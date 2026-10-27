


import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from region_lattice import RegionLattice
import numpy as np

from skimage.io import imread

from adaptive_segmentation import (
    initial_superpixels,
    build_region_graph,
)

image = imread("cat.jpg")[..., :3]

if image.max() > 1:
    image = image.astype(np.float32) / 255.0

labels = initial_superpixels(
    image,
    n_segments=200,
    compactness=10,
)

# Temporary synthetic priority map.
# Replace this with your fuzzy membership-derived score.
H, W = labels.shape
yy, xx = np.mgrid[:H, :W]

cx, cy = W / 2, H / 2
radius = np.sqrt((xx - cx)**2 + (yy - cy)**2)

priority = np.exp(
    -((radius - min(H, W) * 0.25)**2)
    / (2 * (min(H, W) * 0.07)**2)
)

graph = build_region_graph(image, labels, priority)

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(image)
# axes[0].set_title("Original image")

# axes[1].imshow(mark_boundaries(image, labels))
# axes[1].set_title(f"Fine SLIC: {len(np.unique(labels))} regions")

# axes[2].imshow(priority, cmap="magma")
# axes[2].set_title("Refinement priority")

# for ax in axes:
#     ax.axis("off")

# plt.tight_layout()
# plt.show()

print("Initial superpixels:", len(np.unique(labels)))
print("Graph nodes:", graph.number_of_nodes())
print("Graph edges:", graph.number_of_edges())
# image: RGB image
# labels: original SLIC labels
# priority: pixel-level boundary/refinement priority in [0, 1]

lattice = RegionLattice(image, labels, priority)

print("Initial:", lattice.summary())

# Merge similar, low-boundary-priority regions.
num_merges = lattice.merge_low_priority(
    max_cost=0.15,
    max_merges=175,
    appearance_weight=1.0,
    boundary_weight=2.0,
)

print("After merging:", lattice.summary())
print("Merges performed:", num_merges)

merged_labels = lattice.segmentation()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(mark_boundaries(image, labels))
axes[0].set_title("Original SLIC regions")
axes[1].imshow(mark_boundaries(image, merged_labels))
axes[1].set_title("Adaptive merged regions")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
