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
NUM_SUPERPIXELS = 150
SLIC_COMPACTNESS = 10.0
N_ITERATIONS = 5

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


# ---------------------------------------------------------
# 6. Run adaptive inference
# ---------------------------------------------------------

history = []

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

    history.append(log)

    print(
        f"Iteration {iteration + 1}/{N_ITERATIONS}: "
        f"active_regions={len(lattice.active_ids)}, "
        f"merges={len(log['boundary_merges'])}, "
        f"growth_tests={len(log['exterior_growth'])}, "
        f"shrink_tests={len(log['interior_shrink'])}, "
        f"oracle_queries={len(oracle.cache)}"
    )


# ---------------------------------------------------------
# 7. Validate outputs
# ---------------------------------------------------------

assert np.isfinite(memberships).all()
assert np.all(memberships >= 0)
assert np.allclose(
    memberships.sum(axis=-1), 1.0, atol=1e-5
)

final_segmentation = lattice.segmentation()

assert final_segmentation.shape == image.shape[:2]
assert np.all(final_segmentation >= 0)

print("Final active regions:", len(lattice.active_ids))
print("Unique final segments:", len(np.unique(final_segmentation)))
print("Cached oracle masks:", len(oracle.cache))


# ---------------------------------------------------------
# 8. Visualize segmentation and membership maps
# ---------------------------------------------------------

interior_map = memberships[..., 0]
exterior_map = memberships[..., 1]
boundary_map = memberships[..., 2]

# Signed evidence: positive means interior/supporting evidence;
# negative means exterior evidence. Boundary is represented
# separately because it indicates uncertainty/context dependence.
signed_evidence = interior_map - exterior_map

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Input image")

axes[0, 1].imshow(base_labels, cmap="nipy_spectral")
axes[0, 1].set_title("Initial SLIC regions")

axes[0, 2].imshow(final_segmentation, cmap="nipy_spectral")
axes[0, 2].set_title("Final adaptive regions")

im = axes[1, 0].imshow(
    signed_evidence,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
axes[1, 0].set_title("Signed evidence")
fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

im = axes[1, 1].imshow(
    boundary_map,
    cmap="magma",
    vmin=0,
    vmax=1,
)
axes[1, 1].set_title("Boundary membership")
fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

membership_entropy_map = -np.sum(
    memberships * np.log2(np.clip(memberships, 1e-12, 1.0)),
    axis=-1,
) / np.log2(3)

im = axes[1, 2].imshow(
    membership_entropy_map,
    cmap="viridis",
    vmin=0,
    vmax=1,
)
axes[1, 2].set_title("Membership uncertainty")
fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()