import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHECKPOINT_PATH = "best_resnet18_oxford_pet.pth"
IMAGE_PATH = "./data/oxford-iiit-pet/images/Abyssinian_2.jpg"

NUM_CLASSES = 37           # Must match the checkpoint
IMAGE_SIZE = 224
NUM_SUPERPIXELS = 60
SLIC_COMPACTNESS = 10.0
N_ITERATIONS = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# These must match the classifier's training preprocessing.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


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

    # Support common checkpoint conventions.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            "Expected a state-dict checkpoint. "
            "If you saved the complete model object, load it separately."
        )

    # Handle checkpoints saved from DataParallel.
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


# ---------------------------------------------------------
# 2. Load and preprocess the image
# ---------------------------------------------------------

image_pil = Image.open(IMAGE_PATH).convert("RGB")
image_pil = image_pil.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.BILINEAR,
)

# Float RGB image in [0, 1], used by the oracle and lattice.
image = np.asarray(image_pil, dtype=np.float32) / 255.0

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ---------------------------------------------------------
# 3. Model-agnostic probability prediction interface
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


original_probabilities = predict_fn(image)
original_class = int(np.argmax(original_probabilities))

print("Original probabilities:", original_probabilities)
print("Predicted class:", original_class)
print("Original confidence:", float(original_probabilities[original_class]))


# ---------------------------------------------------------
# 4. Initialize SLIC superpixels and boundary prior
# ---------------------------------------------------------

base_labels = slic(
    image,
    n_segments=NUM_SUPERPIXELS,
    compactness=SLIC_COMPACTNESS,
    start_label=0,
    channel_axis=-1,
    enforce_connectivity=True,
)

# Initial boundary prior: high near SLIC boundaries.
boundary_pixels = find_boundaries(
    base_labels,
    mode="thick",
).astype(np.float32)

priority = gaussian_filter(boundary_pixels, sigma=1.5)

if priority.max() > 0:
    priority /= priority.max()

# Initialize memberships: [interior, exterior, boundary].
memberships = initialize_memberships(priority)

# Initialize the dynamic region lattice.
lattice = RegionLattice(
    image=image,
    base_labels=base_labels,
    priority=memberships[..., 2],
)

lattice.update_priority(memberships[..., 2])

print("Initial superpixels:", len(np.unique(base_labels)))
print("Initial active regions:", len(lattice.active_ids))


# ---------------------------------------------------------
# 5. Create the perturbation oracle
# ---------------------------------------------------------

# The default oracle baseline is the per-channel image mean.
# This can be replaced with a blurred or inpainting baseline
# in subsequent robustness experiments.
oracle = PerturbationOracle(
    image=image,
    predict_fn=predict_fn,
)

# Initialize the tracker for conditional perturbation effects.
entropy_tracker = OutcomeEntropyTracker(min_observations=3)



from matplotlib.widgets import Slider


# ---------------------------------------------------------
# 6. Save a snapshot of every iteration
# ---------------------------------------------------------

def capture_snapshot(iteration, image, base_labels, lattice, memberships):
    """Copy all visual data so later updates cannot overwrite it."""
    mu = memberships.copy()

    uncertainty = -np.sum(
        mu * np.log2(np.clip(mu, 1e-12, 1.0)),
        axis=-1,
    ) / np.log2(3)

    return {
        "iteration": iteration,
        "image": image.copy(),
        "base_labels": base_labels.copy(),
        "segmentation": lattice.segmentation().copy(),
        "signed_evidence": (mu[..., 0] - mu[..., 1]).copy(),
        "boundary": mu[..., 2].copy(),
        "uncertainty": uncertainty.copy(),
        "memberships": mu,
    }


snapshots = []

# Iteration 0: before any adaptive updates.
snapshots.append(
    capture_snapshot(0, image, base_labels, lattice, memberships)
)

for iteration in range(N_ITERATIONS):
    log = adaptive_inference_step(
        lattice=lattice,
        oracle=oracle,
        memberships=memberships,
        entropy_tracker=entropy_tracker,
        max_boundary_merges=1,
        max_growth_candidates=2,
        max_shrink_candidates=2,
        rng=np.random.default_rng(1000 + iteration),
    )

    snapshot = capture_snapshot(
        iteration + 1, image, base_labels, lattice, memberships
    )
    snapshots.append(snapshot)

    print(
        f"Iteration {iteration + 1}/{N_ITERATIONS}: "
        f"active_regions={len(lattice.active_ids)}, "
        f"merges={len(log['boundary_merges'])}, "
        f"growth_tests={len(log['exterior_growth'])}, "
        f"shrink_tests={len(log['interior_shrink'])}, "
        f"oracle_queries={len(oracle.cache)}"
    )

# Validate the final memberships.
assert np.isfinite(memberships).all()
assert np.all(memberships >= 0)
assert np.allclose(memberships.sum(axis=-1), 1.0, atol=1e-5)


# ---------------------------------------------------------
# 7. Interactive Matplotlib viewer
# ---------------------------------------------------------

def show_iteration_viewer(snapshots):
    """
    Browse all saved iterations with a slider.
    Requires a Matplotlib GUI backend when run as a Python script.
    """
    first = snapshots[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(
        left=0.035, right=0.96,
        top=0.91, bottom=0.18,
        wspace=0.22, hspace=0.25,
    )

    ax_input, ax_slic, ax_regions = axes[0]
    ax_signed, ax_boundary, ax_uncertainty = axes[1]

    # Static image panels.
    ax_input.imshow(first["image"])
    ax_input.set_title("Input image")

    ax_slic.imshow(
        first["base_labels"], cmap="nipy_spectral",
        interpolation="nearest",
    )
    ax_slic.set_title("Initial SLIC regions")

    # Dynamic panels.
    region_im = ax_regions.imshow(
        first["segmentation"], cmap="nipy_spectral",
        interpolation="nearest",
    )
    ax_regions.set_title("Adaptive regions — iteration 0")

    signed_im = ax_signed.imshow(
        first["signed_evidence"],
        cmap="coolwarm", vmin=-1, vmax=1,
    )
    ax_signed.set_title("Signed evidence")
    fig.colorbar(signed_im, ax=ax_signed, fraction=0.046, pad=0.04)

    boundary_im = ax_boundary.imshow(
        first["boundary"],
        cmap="magma", vmin=0, vmax=1,
    )
    ax_boundary.set_title("Boundary membership")
    fig.colorbar(boundary_im, ax=ax_boundary, fraction=0.046, pad=0.04)

    uncertainty_im = ax_uncertainty.imshow(
        first["uncertainty"],
        cmap="viridis", vmin=0, vmax=1,
    )
    ax_uncertainty.set_title("Membership uncertainty")
    fig.colorbar(
        uncertainty_im, ax=ax_uncertainty,
        fraction=0.046, pad=0.04,
    )

    for ax in axes.flat:
        ax.axis("off")

    # Slider axis.
    slider_ax = fig.add_axes([0.18, 0.075, 0.64, 0.035])

    iteration_slider = Slider(
        ax=slider_ax,
        label="Iteration",
        valmin=0,
        valmax=len(snapshots) - 1,
        valinit=0,
        valstep=1,
        valfmt="%0.0f",
    )

    def update(value):
        idx = int(value)
        snap = snapshots[idx]

        region_im.set_data(snap["segmentation"])
        signed_im.set_data(snap["signed_evidence"])
        boundary_im.set_data(snap["boundary"])
        uncertainty_im.set_data(snap["uncertainty"])

        ax_regions.set_title(
            f"Adaptive regions — iteration {snap['iteration']}"
        )

        fig.suptitle(
            f"Adaptive segmentation progress "
            f"({idx + 1}/{len(snapshots)} snapshots)",
            fontsize=14,
        )

        fig.canvas.draw_idle()

    iteration_slider.on_changed(update)

    fig.suptitle("Adaptive segmentation progress", fontsize=14)
    plt.show()


show_iteration_viewer(snapshots)
