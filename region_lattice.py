
from dataclasses import dataclass, field

import numpy as np
from skimage.color import rgb2lab


@dataclass
class RegionNode:
    region_id: int
    leaves: frozenset
    mean_lab: np.ndarray
    area: int
    mean_priority: float
    q90_priority: float

    # Alternative ways to decompose this region into two children.
    decompositions: list = field(default_factory=list)


class RegionLattice:
    """
    Sparse lattice of connected image regions.

    Base SLIC labels remain immutable. The current segmentation is a
    separate partition, while merged regions and their decompositions
    persist across changes in refinement priority.
    """

    def __init__(self, image, base_labels, priority):
        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        self.image = image
        self.lab = rgb2lab(image)
        self.base_labels = np.asarray(base_labels, dtype=np.int32).copy()

        self._check_shapes(priority)
        self.priority = np.clip(
            np.asarray(priority, dtype=np.float32), 0.0, 1.0
        )

        self.nodes = {}
        self.leaves_to_id = {}
        self.next_id = int(self.base_labels.max()) + 1

        # Initially, each SLIC label is its own region, with the same ID.
        self.current_map = self.base_labels.copy()

        for label in np.unique(self.base_labels):
            label = int(label)
            mask = self.base_labels == label
            values = self.priority[mask]

            self.nodes[label] = RegionNode(
                region_id=label,
                leaves=frozenset([label]),
                mean_lab=self.lab[mask].mean(axis=0),
                area=int(mask.sum()),
                mean_priority=float(values.mean()),
                q90_priority=float(np.quantile(values, 0.90)),
            )
            self.leaves_to_id[frozenset([label])] = label

    def _check_shapes(self, priority):
        if self.base_labels.shape != self.image.shape[:2]:
            raise ValueError("base_labels must match the image dimensions")
        if np.asarray(priority).shape != self.base_labels.shape:
            raise ValueError("priority must match the label dimensions")

    def _get_or_create_node(self, leaves, children=None):
        """Return the canonical node for a set of base superpixels."""
        leaves = frozenset(leaves)

        if leaves in self.leaves_to_id:
            region_id = self.leaves_to_id[leaves]
            node = self.nodes[region_id]

            if children is not None and children not in node.decompositions:
                node.decompositions.append(children)

            return region_id

        mask = np.isin(self.base_labels, list(leaves))
        values = self.priority[mask]

        region_id = self.next_id
        self.next_id += 1

        node = RegionNode(
            region_id=region_id,
            leaves=leaves,
            mean_lab=self.lab[mask].mean(axis=0),
            area=int(mask.sum()),
            mean_priority=float(values.mean()),
            q90_priority=float(np.quantile(values, 0.90)),
        )

        if children is not None:
            node.decompositions.append(children)

        self.nodes[region_id] = node
        self.leaves_to_id[leaves] = region_id
        return region_id

    @property
    def active_ids(self):
        """IDs in the current segmentation, not all historical nodes."""
        return set(np.unique(self.current_map).tolist())

    def adjacency(self):
        """Return pairs of regions sharing a pixel boundary."""
        pairs = []

        for a, b in (
            (self.current_map[:, :-1], self.current_map[:, 1:]),
            (self.current_map[:-1, :], self.current_map[1:, :]),
        ):
            different = a != b
            if np.any(different):
                p = np.stack([a[different], b[different]], axis=1)
                p = np.sort(p, axis=1)
                pairs.append(np.unique(p, axis=0))

        if not pairs:
            return set()

        all_pairs = np.unique(np.concatenate(pairs, axis=0), axis=0)
        return {tuple(map(int, pair)) for pair in all_pairs}

    
    def merge_cost(
        self,
        u,
        v,
        appearance_weight=1.0,
        boundary_weight=2.0,
    ):
        if u not in self.nodes or v not in self.nodes:
            raise KeyError(
                f"Unknown region ID: u={u}, v={v}. "
                f"Stored IDs: {list(self.nodes.keys())}; "
                f"Active IDs: {sorted(self.active_ids)}"
            )

        a, b = self.nodes[u], self.nodes[v]

        color_distance = (
            np.linalg.norm(a.mean_lab - b.mean_lab) / 100.0
        )
        boundary_priority = max(a.q90_priority, b.q90_priority)

        return (
            appearance_weight * color_distance
            + boundary_weight * boundary_priority
        )


    def merge(self, u, v):
        """Merge adjacent active regions and preserve their decomposition."""
        if u not in self.active_ids or v not in self.active_ids:
            raise ValueError("Both regions must be active")

        if u == v or tuple(sorted((u, v))) not in self.adjacency():
            raise ValueError("Regions must be distinct and adjacent")

        leaves_u = self.nodes[u].leaves
        leaves_v = self.nodes[v].leaves

        if leaves_u & leaves_v:
            raise RuntimeError("Active regions must have disjoint leaves")

        new_id = self._get_or_create_node(
            leaves_u | leaves_v,
            children=(u, v),
        )

        # Replace the two regions in the current partition.
        mask = (self.current_map == u) | (self.current_map == v)
        self.current_map[mask] = new_id

        return new_id

    def merge_low_priority(
        self,
        max_cost,
        max_merges=None,
        appearance_weight=1.0,
        boundary_weight=2.0,
    ):
        """Greedily merge the lowest-cost adjacent pair until stopped."""
        merges = 0

        while max_merges is None or merges < max_merges:
            candidates = [
                (
                    self.merge_cost(
                        u, v,
                        appearance_weight=appearance_weight,
                        boundary_weight=boundary_weight,
                    ),
                    u,
                    v,
                )
                for u, v in self.adjacency()
            ]

            if not candidates:
                break

            cost, u, v = min(candidates)

            if cost > max_cost:
                break

            self.merge(u, v)
            merges += 1

        return merges

    def split(self, region_id, decomposition=-1):
        """
        Revert one active region to a previously recorded pair of children.
        Other current regions remain unchanged.
        """
        if region_id not in self.active_ids:
            raise ValueError("Region is not active")

        node = self.nodes[region_id]
        if not node.decompositions:
            raise ValueError("No recorded decomposition for this region")

        u, v = node.decompositions[decomposition]
        parent_mask = self.current_map == region_id

        mask_u = parent_mask & np.isin(
            self.base_labels, list(self.nodes[u].leaves)
        )
        mask_v = parent_mask & np.isin(
            self.base_labels, list(self.nodes[v].leaves)
        )

        if not np.all(mask_u | mask_v == parent_mask):
            raise RuntimeError("Stored children do not cover the parent")

        self.current_map[mask_u] = u
        self.current_map[mask_v] = v

        return u, v

    def update_priority(self, new_priority):
        """
        Replace the pixel-level refinement priority and refresh statistics
        for all stored nodes. The current partition is preserved.
        """
        new_priority = np.asarray(new_priority, dtype=np.float32)
        self._check_shapes(new_priority)

        self.priority = np.clip(new_priority, 0.0, 1.0)

        for node in self.nodes.values():
            mask = np.isin(
                self.base_labels, list(node.leaves)
            )
            values = self.priority[mask]
            node.mean_priority = float(values.mean())
            node.q90_priority = float(np.quantile(values, 0.90))

    def segmentation(self):
        """Return a compact label map for the current partition."""
        active = sorted(self.active_ids)
        result = np.zeros(self.current_map.shape, dtype=np.int32)

        for compact_id, region_id in enumerate(active):
            result[self.current_map == region_id] = compact_id

        return result

    def summary(self):
        return {
            "active_regions": len(self.active_ids),
            "stored_nodes": len(self.nodes),
            "active_boundaries": len(self.adjacency()),
        }
