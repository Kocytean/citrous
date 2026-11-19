
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import gaussian_filter

from adaptive_segmentation import (
    PerturbationOracle,
    OutcomeEntropyTracker,
    initialize_memberships,
    adaptive_inference_step,
)
from region_lattice import RegionLattice


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHECKPOINT_PATH = "best_resnet18_oxford_pet.pth"

# Add more image paths here to calculate dataset averages.
from pathlib import Path

IMAGE_DIR = Path("./data/oxford-iiit-pet/images")

IMAGE_PATHS = sorted(
    list(IMAGE_DIR.glob("*.jpg"))
)[:30]

# Oxford-IIIT Pet annotation directory.
TRIMAP_DIR = Path("./data/oxford-iiit-pet/annotations/trimaps")

NUM_CLASSES = 37
IMAGE_SIZE = 224
NUM_SUPERPIXELS = 150
SLIC_COMPACTNESS = 10.0
N_ITERATIONS = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Oxford-IIIT Pet trimap labels:
# 1 = foreground, 2 = background, 3 = boundary.
# Include boundary pixels in the complete ground-truth pet mask.
GROUND_TRUTH_FOREGROUND_LABELS = (1, 3)

# Adaptive inference settings.
MAX_BOUNDARY_MERGES = 1
MAX_GROWTH_CANDIDATES = 2
MAX_SHRINK_CANDIDATES = 2


# ---------------------------------------------------------
# 1. Load the trained ResNet18
# ---------------------------------------------------------

def load_model(checkpoint_path, num_classes, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            "Expected a state-dict checkpoint."
        )

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


model = load_model(CHECKPOINT_PATH, NUM_CLASSES, DEVICE)
print(f"Loaded model on {DEVICE}")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ---------------------------------------------------------
# 2. Model-agnostic probability prediction interface
# ---------------------------------------------------------

@torch.inference_mode()
def predict_fn(image_rgb):
    """
    Input:
        RGB numpy image, shape (H, W, 3), range [0, 1].

    Output:
        Class probabilities, shape (NUM_CLASSES,).
    """
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    image_rgb = np.clip(image_rgb, 0.0, 1.0)

    pil_image = Image.fromarray(
        np.uint8(np.round(image_rgb * 255.0))
    )

    x = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probabilities = F.softmax(logits, dim=1)

    return probabilities[0].cpu().numpy()


# ---------------------------------------------------------
# 3. Load image and ground-truth trimap
# ---------------------------------------------------------

def load_image_and_mask(image_path):
    """
    Loads an Oxford-IIIT Pet image and its corresponding trimap.

    Returns:
        image:   RGB float image in [0, 1], shape (H, W, 3)
        gt_mask: binary foreground mask, shape (H, W)
    """
    image_path = Path(image_path)

    image_pil = Image.open(image_path).convert("RGB")
    image_pil = image_pil.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    image = np.asarray(
        image_pil,
        dtype=np.float32
    ) / 255.0

    # The trimap filename has the same stem as the image.
    trimap_path = TRIMAP_DIR / f"{image_path.stem}.png"

    if not trimap_path.exists():
        raise FileNotFoundError(
            f"Ground-truth trimap not found: {trimap_path}"
        )

    trimap = Image.open(trimap_path)

    # Nearest-neighbor interpolation preserves integer labels.
    trimap = trimap.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST,
    )

    trimap = np.asarray(trimap)

    gt_mask = np.isin(
        trimap,
        GROUND_TRUTH_FOREGROUND_LABELS,
    )

    if gt_mask.sum() == 0:
        raise ValueError(
            f"Ground-truth foreground is empty: {trimap_path}"
        )

    return image, gt_mask


# ---------------------------------------------------------
# 4. Segmentation metrics
# ---------------------------------------------------------
def calculate_segmentation_metrics(memberships, gt_mask):

    if not np.isfinite(memberships).all():
        raise ValueError(
            "Memberships contain NaN or Inf. "
            "Check membership updates before calculating metrics."
        )

    if not np.isfinite(gt_mask).all():
        raise ValueError("Ground-truth mask contains NaN or Inf.")

    predicted_labels = np.argmax(memberships, axis=-1)

    # interior = predicted_labels == 0
    # interior_boundary = (
    #     (predicted_labels == 0) |
    #     (predicted_labels == 2)
    # )
    interior = memberships[..., 0] > 0.5

    interior_boundary = (
        (memberships[..., 0] + memberships[..., 2]) > 0.5
    )
    gt = gt_mask.astype(bool)

    interior_intersection = int(
        np.logical_and(interior, gt).sum()
    ) / (int(interior.sum()) +0.0001)

    combined_intersection = np.logical_and(
        interior_boundary, gt
    ).sum()

    combined_union = np.logical_or(
        interior_boundary, gt
    ).sum() +0.0001

    combined_iou = (
        float(combined_intersection / combined_union)
        if combined_union > 0 else 0.0
    )

    return {
        "interior_intersection": interior_intersection,
        "interior_boundary_iou": combined_iou,
    }
# ---------------------------------------------------------
# 5. Run adaptive inference on one image
# ---------------------------------------------------------

def run_adaptive_inference(image_path):
    image, gt_mask = load_image_and_mask(image_path)

    # Original classifier prediction.
    original_probabilities = predict_fn(image)
    original_class = int(np.argmax(original_probabilities))

    print(f"\nImage: {Path(image_path).name}")
    print("Predicted class:", original_class)
    print(
        "Original confidence:",
        float(original_probabilities[original_class]),
    )

    # -----------------------------------------------------
    # Initialize SLIC superpixels
    # -----------------------------------------------------

    base_labels = slic(
        image,
        n_segments=NUM_SUPERPIXELS,
        compactness=SLIC_COMPACTNESS,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=True,
    )

    # Initial boundary prior.
    boundary_pixels = find_boundaries(
        base_labels,
        mode="thick",
    ).astype(np.float32)

    priority = gaussian_filter(
        boundary_pixels,
        sigma=1.5,
    )

    if priority.max() > 0:
        priority /= priority.max()

    # Initialize memberships.
    memberships = initialize_memberships(priority)

    # Initialize region lattice.
    lattice = RegionLattice(
        image=image,
        base_labels=base_labels,
        priority=memberships[..., 2],
    )

    lattice.update_priority(memberships[..., 2])

    print("Initial superpixels:", len(np.unique(base_labels)))
    print("Initial active regions:", len(lattice.active_ids))

    # -----------------------------------------------------
    # Perturbation oracle and entropy tracker
    # -----------------------------------------------------

    oracle = PerturbationOracle(
        image=image,
        predict_fn=predict_fn,
    )

    entropy_tracker = OutcomeEntropyTracker(
        min_observations=3
    )

    # -----------------------------------------------------
    # Record metrics at iteration 0 (initial SLIC state)
    # -----------------------------------------------------

    metric_history = {
    "interior_intersection": [],
    "interior_boundary_iou": []}

    initial_metrics = calculate_segmentation_metrics(
        memberships,
        gt_mask,
    )

    for key in metric_history:
        metric_history[key].append(initial_metrics[key])

    print(
        "Iteration 0: "
        f"Interior intersection={initial_metrics["interior_intersection"]:.4f}, "
        f"Interior+Boundary IoU="
        f"{initial_metrics['interior_boundary_iou']:.4f}"
    )

    # -----------------------------------------------------
    # Adaptive inference
    # -----------------------------------------------------

    history = []

    for iteration in range(N_ITERATIONS):

        log = adaptive_inference_step(
            lattice=lattice,
            oracle=oracle,
            memberships=memberships,
            entropy_tracker=entropy_tracker,
            max_boundary_merges=MAX_BOUNDARY_MERGES,
            max_growth_candidates=MAX_GROWTH_CANDIDATES,
            max_shrink_candidates=MAX_SHRINK_CANDIDATES,
            rng=np.random.default_rng(1000 + iteration),
        )

        history.append(log)

        # Calculate metrics after this iteration.
        metrics = calculate_segmentation_metrics(
            memberships,
            gt_mask,
        )

        for key in metric_history:
            metric_history[key].append(metrics[key])

        print(
            f"Iteration {iteration + 1}/{N_ITERATIONS}: "
            f"active_regions={len(lattice.active_ids)}, "
            f"merges={len(log['boundary_merges'])}, "
            f"growth_tests={len(log['exterior_growth'])}, "
            f"shrink_tests={len(log['interior_shrink'])}, "
            f"oracle_queries={len(oracle.cache)}"
        )

        print(
        "Iteration 0: "
        f"Interior intersection={metrics["interior_intersection"]:.4f}, "
        f"Interior+Boundary IoU="
        f"{metrics['interior_boundary_iou']:.4f}"
    )

    # -----------------------------------------------------
    # Validate outputs
    # -----------------------------------------------------

    assert np.isfinite(memberships).all()
    assert np.all(memberships >= 0)
    assert np.allclose(
        memberships.sum(axis=-1),
        1.0,
        atol=1e-5,
    )

    final_segmentation = lattice.segmentation()

    assert final_segmentation.shape == image.shape[:2]
    assert np.all(final_segmentation >= 0)

    print("Final active regions:", len(lattice.active_ids))
    print("Unique final segments:", len(np.unique(final_segmentation)))
    print("Cached oracle masks:", len(oracle.cache))

    # Return results for dataset aggregation and visualization.
    return {
        "image": image,
        "gt_mask": gt_mask,
        "base_labels": base_labels,
        "final_segmentation": final_segmentation,
        "memberships": memberships,
        "metric_history": metric_history,
        "history": history,
    }



# ---------------------------------------------------------
# 6. Evaluate the test dataset
# ---------------------------------------------------------

dataset_metrics = {
    "interior_intersection": [],
    "interior_boundary_iou": [],
}

first_image_result = None

for image_index, image_path in enumerate(IMAGE_PATHS):

    print(
        f"\nProcessing image {image_index + 1}/{len(IMAGE_PATHS)}: "
        f"{image_path.name}"
    )

    result = run_adaptive_inference(image_path)

    if first_image_result is None:
        first_image_result = result

    for key in dataset_metrics:
        dataset_metrics[key].append(
            result["metric_history"][key]
        )


# ---------------------------------------------------------
# 7. Dataset averages and standard deviations
# ---------------------------------------------------------

# Shape: (number_of_images, N_ITERATIONS + 1)
iteration_numbers = np.arange(N_ITERATIONS + 1)
mean_metrics = {
    key: np.mean(values, axis=0)
    for key, values in dataset_metrics.items()
}

std_metrics = {
    key: np.std(values, axis=0)
    for key, values in dataset_metrics.items()
}

iterations = np.arange(N_ITERATIONS + 1)

print("\n========================================")
print("TEST DATASET RESULTS")
print("========================================")

for i in iterations:
    print(
        f"Iteration {i}: "
        f"Interior intersection = "
        f"{mean_metrics['interior_intersection'][i]:.2f} "
        f"(±{std_metrics['interior_intersection'][i]:.2f}), "
        f"Interior + boundary IoU = "
        f"{mean_metrics['interior_boundary_iou'][i]:.4f} "
        f"(±{std_metrics['interior_boundary_iou'][i]:.4f})"
    )


# ---------------------------------------------------------
# 8. Identify best iteration
# ---------------------------------------------------------

best_iteration = int(
    np.argmax(mean_metrics["interior_boundary_iou"])
)

print("\nBest iteration:", best_iteration)
print(
    "Mean interior + boundary IoU:",
    mean_metrics["interior_boundary_iou"][best_iteration]
)


# ---------------------------------------------------------
# 9. Plot dataset averages
# ---------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(
    iterations,
    mean_metrics["interior_intersection"],
    marker="o",
    label="Interior intersection",
)

ax1.fill_between(
    iterations,
    mean_metrics["interior_intersection"]
    - std_metrics["interior_intersection"],
    mean_metrics["interior_intersection"]
    + std_metrics["interior_intersection"],
    alpha=0.15,
)

ax1.set_xlabel("Iteration")
ax1.set_ylabel("Interior intersection (pixels)")
ax1.set_xticks(iterations)
ax1.grid(True)

ax2 = ax1.twinx()

ax2.plot(
    iterations,
    mean_metrics["interior_boundary_iou"],
    marker="s",
    label="Interior + boundary IoU",
)

ax2.fill_between(
    iterations,
    np.clip(
        mean_metrics["interior_boundary_iou"]
        - std_metrics["interior_boundary_iou"],
        0, 1
    ),
    np.clip(
        mean_metrics["interior_boundary_iou"]
        + std_metrics["interior_boundary_iou"],
        0, 1
    ),
    alpha=0.15,
)

ax2.set_ylabel("Interior + boundary IoU")
ax2.set_ylim(0, 1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best",
)

plt.title(
    f"Average Segmentation Performance "
    f"over {len(IMAGE_PATHS)} Test Images"
)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. Plot average metrics across iterations
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

metric_labels = {
    "interior_intersection": "Interior-GT Intersection (pixels)",
    "interior_boundary_iou": "Interior + Boundary IoU",
}

for key, label in metric_labels.items():

    mean = mean_metrics[key]
    std = std_metrics[key]

    plt.plot(
        iteration_numbers,
        mean,
        marker="o",
        label=label,
    )

    plt.fill_between(
        iteration_numbers,
        np.clip(mean - std, 0, 1),
        np.clip(mean + std, 0, 1),
        alpha=0.15,
    )

plt.axvline(
    best_iteration,
    linestyle="--",
    alpha=0.7,
    label=f"Best iteration: {best_iteration}",
)

plt.xlabel("Iteration")
plt.ylabel("Metric")
plt.title(
    f"Average Segmentation Performance "
    f"({len(IMAGE_PATHS)} images)"
)
plt.xticks(iteration_numbers)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 10. Visualize the first image's final segmentation
# ---------------------------------------------------------

# result = first_image_result

# image = result["image"]
# gt_mask = result["gt_mask"]
# base_labels = result["base_labels"]
# final_segmentation = result["final_segmentation"]
# memberships = result["memberships"]

# interior_map = memberships[..., 0]
# exterior_map = memberships[..., 1]
# boundary_map = memberships[..., 2]

# signed_evidence = interior_map - exterior_map

# membership_entropy_map = -np.sum(
#     memberships * np.log2(
#         np.clip(memberships, 1e-12, 1.0)
#     ),
#     axis=-1,
# ) / np.log2(3)

# fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# axes[0, 0].imshow(image)
# axes[0, 0].set_title("Input image")

# axes[0, 1].imshow(gt_mask, cmap="gray")
# axes[0, 1].set_title("Ground-truth foreground")

# axes[0, 2].imshow(base_labels, cmap="nipy_spectral")
# axes[0, 2].set_title("Initial SLIC regions")

# axes[0, 3].imshow(final_segmentation, cmap="nipy_spectral")
# axes[0, 3].set_title("Final adaptive regions")

# im = axes[1, 0].imshow(
#     signed_evidence,
#     cmap="coolwarm",
#     vmin=-1,
#     vmax=1,
# )
# axes[1, 0].set_title("Signed evidence")
# fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

# im = axes[1, 1].imshow(
#     interior_map,
#     cmap="Greens",
#     vmin=0,
#     vmax=1,
# )
# axes[1, 1].set_title("Interior membership")
# fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

# im = axes[1, 2].imshow(
#     boundary_map,
#     cmap="magma",
#     vmin=0,
#     vmax=1,
# )
# axes[1, 2].set_title("Boundary membership")
# fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

# im = axes[1, 3].imshow(
#     membership_entropy_map,
#     cmap="viridis",
#     vmin=0,
#     vmax=1,
# )
# axes[1, 3].set_title("Membership uncertainty")
# fig.colorbar(im, ax=axes[1, 3], fraction=0.046)

# for ax in axes.flat:
#     ax.axis("off")

# plt.tight_layout()
# plt.show()
