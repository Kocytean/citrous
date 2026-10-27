
import numpy as np
import networkx as nx

from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.measure import regionprops


def initial_superpixels(
    image,
    n_segments=400,
    compactness=10.0,
    sigma=1.0,
):
    """
    Generate fine initial superpixels.

    Parameters
    ----------
    image : ndarray, H x W x 3
        RGB image, with values in [0, 1] or [0, 255].
    n_segments : int
        Approximate number of initial superpixels.
    compactness : float
        SLIC trade-off between spatial regularity and color similarity.
    sigma : float
        Gaussian smoothing applied by SLIC.

    Returns
    -------
    labels : ndarray, H x W
        Integer superpixel labels.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    labels = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
        convert2lab=True,
    )

    return labels.astype(np.int32)


def build_region_graph(image, labels, priority):
    """
    Construct a region adjacency graph.

    Nodes represent superpixels.
    Edges connect spatially adjacent superpixels.

    Each node stores:
      - area
      - mean Lab color
      - mean refinement priority

    Parameters
    ----------
    image : ndarray, H x W x 3
        RGB image.
    labels : ndarray, H x W
        Superpixel labels.
    priority : ndarray, H x W
        Higher values mean the region should remain finer.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    lab = rgb2lab(image)
    priority = np.asarray(priority, dtype=np.float32)

    if priority.shape != labels.shape:
        raise ValueError("priority and labels must have the same shape")

    graph = nx.Graph()

    for region in regionprops(labels + 1):
        label = region.label - 1
        mask = labels == label

        graph.add_node(
            label,
            area=int(mask.sum()),
            mean_lab=lab[mask].mean(axis=0),
            priority=float(priority[mask].mean()),
            max_priority=float(priority[mask].max()),
        )

    # Connect labels that share a horizontal or vertical boundary.
    for a, b in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        different = a != b

        pairs = np.stack(
            [a[different], b[different]], axis=1
        )

        if len(pairs) == 0:
            continue

        pairs = np.sort(pairs, axis=1)
        pairs = np.unique(pairs, axis=0)

        for u, v in pairs:
            graph.add_edge(int(u), int(v))

    return graph


def merge_cost(graph, u, v, appearance_weight=1.0,
               priority_weight=2.0):
    """
    Cost of merging adjacent regions.

    Similar appearance lowers the cost.
    High refinement priority raises the cost.
    """
    a = graph.nodes[u]
    b = graph.nodes[v]

    color_distance = np.linalg.norm(
        a["mean_lab"] - b["mean_lab"]
    ) / 100.0

    priority = 0.5 * (
        a["priority"] + b["priority"]
    )

    return (
        appearance_weight * color_distance
        + priority_weight * priority
    )


def adaptive_merge(labels, graph, max_merge_cost=0.15):
    """
    Greedily merge adjacent low-priority regions.

    The returned segmentation is a flat partition.
    The original fine labels are not modified.

    This is a first baseline; the next iteration will add
    explicit parent-child hierarchy tracking and splitting.
    """
    graph = graph.copy()

    while graph.number_of_edges() > 0:
        candidates = [
            (merge_cost(graph, u, v), u, v)
            for u, v in graph.edges()
        ]

        cost, u, v = min(candidates)

        if cost > max_merge_cost:
            break

        # Preserve area-weighted color and priority statistics.
        a = graph.nodes[u]
        b = graph.nodes[v]
        area_a, area_b = a["area"], b["area"]
        total_area = area_a + area_b

        merged_lab = (
            area_a * a["mean_lab"]
            + area_b * b["mean_lab"]
        ) / total_area

        merged_priority = (
            area_a * a["priority"]
            + area_b * b["priority"]
        ) / total_area

        merged_max_priority = max(
            a["max_priority"], b["max_priority"]
        )

        # Use a new node ID so the merge history remains
        # unambiguous within this graph operation.
        new_id = max(graph.nodes, default=-1) + 1

        neighbors = (
            set(graph.neighbors(u))
            | set(graph.neighbors(v))
        ) - {u, v}

        graph.add_node(
            new_id,
            area=total_area,
            mean_lab=merged_lab,
            priority=merged_priority,
            max_priority=merged_max_priority,
        )

        for neighbor in neighbors:
            graph.add_edge(new_id, neighbor)

        graph.remove_nodes_from([u, v])

    return graph


class PerturbationOracle:
    """
    Black-box classifier interface.

    predict_fn(image) must return a 1D vector of class probabilities
    for one HWC image. Images passed to this class are float RGB in [0, 1].
    """

    def __init__(self, image, predict_fn, baseline=None):
        self.image = np.asarray(image, dtype=np.float32)
        self.predict_fn = predict_fn

        if baseline is None:
            baseline = self.image.reshape(-1, 3).mean(axis=0)

        self.baseline = np.asarray(baseline, dtype=np.float32)
        self.original_probs = np.asarray(
            predict_fn(self.image), dtype=np.float32
        )
        self.target_class = int(np.argmax(self.original_probs))
        self.target_score = float(
            self.original_probs[self.target_class]
        )
        self.cache = {}

    def evaluate(self, mask):
        """
        mask=True means replace those pixels with the baseline.
        Returns cached prediction and evidence if already evaluated.
        """
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != self.image.shape[:2]:
            raise ValueError("Mask must match image height and width")

        # Pack the binary mask into a compact, deterministic cache key.
        key = np.packbits(mask.reshape(-1)).tobytes()

        if key in self.cache:
            return self.cache[key]

        perturbed = self.image.copy()
        perturbed[mask] = self.baseline

        probs = np.asarray(
            self.predict_fn(perturbed), dtype=np.float32
        )

        pred_class = int(np.argmax(probs))
        changed = pred_class != self.target_class
        score_drop = self.target_score - float(
            probs[self.target_class]
        )

        result = {
            "changed": changed,
            "prediction": pred_class,
            "probabilities": probs,
            "score_drop": score_drop,
            "mask_fraction": float(mask.mean()),
        }

        self.cache[key] = result
        return result

def initialize_memberships(priority):
    """
    priority is a pixel-level boundary score in [0, 1].
    Returns memberships with shape (H, W, 3).
    """
    boundary = np.clip(
        np.asarray(priority, dtype=np.float32), 0.0, 1.0
    )

    interior_exterior = 1.0 - boundary

    # Initially split non-boundary membership equally.
    memberships = np.stack(
        [
            0.5 * interior_exterior,
            0.5 * interior_exterior,
            boundary,
        ],
        axis=-1,
    )

    return memberships


def update_membership(memberships, mask, category, alpha=0.25):
    """
    Update all pixels in mask toward a category:
      0 = interior, 1 = exterior, 2 = boundary.
    """
    if category not in (0, 1, 2):
        raise ValueError("category must be 0, 1, or 2")

    mask = np.asarray(mask, dtype=bool)
    target = np.zeros(3, dtype=np.float32)
    target[category] = 1.0

    memberships[mask] = (
        (1.0 - alpha) * memberships[mask]
        + alpha * target
    )

    memberships[mask] /= np.maximum(
        memberships[mask].sum(axis=-1, keepdims=True), 1e-8
    )

    return memberships


def membership_entropy(memberships, mask, eps=1e-8):
    """Mean categorical entropy over a region."""
    values = memberships[mask]

    if values.size == 0:
        return 0.0

    entropy = -np.sum(
        values * np.log(values + eps), axis=-1
    )

    return float(entropy.mean())

def region_mask(lattice, region_id):
    """Pixel mask for a stored region, including merged regions."""
    leaves = lattice.nodes[region_id].leaves
    return np.isin(lattice.base_labels, list(leaves))


def sample_neighbors(
    lattice,
    region_id,
    memberships,
    entropy_tracker,
    count=3,
    rng=None,
    boundary_weight=0.2,
    membership_entropy_weight=0.4,
    outcome_entropy_weight=0.4,
):
    rng = np.random.default_rng() if rng is None else rng

    adjacency = lattice.adjacency()
    neighbors = set()

    for u, v in adjacency:
        if u == region_id:
            neighbors.add(v)
        elif v == region_id:
            neighbors.add(u)

    neighbors = [
        rid for rid in neighbors
        if rid in lattice.active_ids
    ]

    if not neighbors:
        return []

    scores = []

    for rid in neighbors:
        mask = region_mask(lattice, rid)

        mu = memberships[mask].mean(axis=0)
        h_membership = membership_entropy(
            memberships, mask
        ) / np.log2(3)
        h_membership = float(np.clip(h_membership, 0.0, 1.0))

        h_outcome = entropy_tracker.priority_entropy(rid)
        boundary = float(mu[2])

        score = (
            membership_entropy_weight * h_membership
            + outcome_entropy_weight * h_outcome
            + boundary_weight * boundary
        )

        scores.append(max(float(score), 1e-8))

    probabilities = np.asarray(scores, dtype=float)
    probabilities /= probabilities.sum()

    k = min(count, len(neighbors))
    indices = rng.choice(
        len(neighbors),
        size=k,
        replace=False,
        p=probabilities,
    )

    return [neighbors[int(i)] for i in indices]

def grow_exterior(
    lattice,
    oracle,
    memberships,
    exterior_id,
    entropy_tracker,
    count=3,
    alpha=0.25,
    min_score_drop=0.05,
    rng=None,
):
    """
    Grow an exterior region by merging contiguous neighbors
    whose masking has sufficiently weak target-class support.

    score_drop = original target probability
                 - perturbed target probability

    Positive conditional score drop means that masking the
    neighbor reduces target-class confidence further.
    """

    rng = np.random.default_rng() if rng is None else rng
    results = []

    if exterior_id not in lattice.active_ids:
        return [{
            "status": "inactive_seed",
            "exterior_id": exterior_id,
        }]

    for attempt in range(count):
        if exterior_id not in lattice.active_ids:
            break

        # Re-evaluate the current, potentially enlarged exterior.
        exterior_mask = region_mask(lattice, exterior_id)
        base_outcome = oracle.evaluate(exterior_mask)

        # If masking the seed changes the class, do not grow it
        # as exterior. Its current role needs reassessment.
        if base_outcome["changed"]:
            update_membership(
                memberships, exterior_mask, 2, alpha
            )
            lattice.update_priority(memberships[..., 2])

            results.append({
                "exterior_id": exterior_id,
                "status": "seed_reclassified_as_boundary",
                "outcome": base_outcome,
            })
            break

        # Sample ONE neighbor at a time because every merge
        # changes the active graph and invalidates old candidates.
        sampled = sample_neighbors(
            lattice=lattice,
            region_id=exterior_id,
            memberships=memberships,
            entropy_tracker=entropy_tracker,
            count=1,
            rng=rng,
        )

        if not sampled:
            results.append({
                "exterior_id": exterior_id,
                "status": "no_active_neighbors",
            })
            break

        neighbor_id = sampled[0]

        if neighbor_id not in lattice.active_ids:
            continue

        neighbor_mask = region_mask(lattice, neighbor_id)
        combined_mask = exterior_mask | neighbor_mask

        combined_outcome = oracle.evaluate(combined_mask)

        # Conditional change relative to masking the exterior alone.
        conditional_drop = (
            float(combined_outcome["score_drop"])
            - float(base_outcome["score_drop"])
        )

        # Record whether adding this neighbor changes the decision
        # relative to the current exterior-mask context.
        entropy_tracker.record(
            neighbor_id,
            combined_outcome["changed"] != base_outcome["changed"],
        )

        # Test the neighbor by itself only when the combined test
        # indicates that it may be important or context-dependent.
        individual_outcome = None

        if combined_outcome["changed"]:
            individual_outcome = oracle.evaluate(neighbor_mask)

            if individual_outcome["changed"]:
                category = 0
                reason = "neighbor_has_individual_supporting_evidence"
            else:
                category = 2
                reason = "combined_mask_changes_decision_contextually"

            update_membership(
                memberships, neighbor_mask, category, alpha
            )

            results.append({
                "exterior_id": exterior_id,
                "neighbor_id": neighbor_id,
                "status": "not_merged",
                "reason": reason,
                "base_outcome": base_outcome,
                "combined_outcome": combined_outcome,
                "individual_outcome": individual_outcome,
                "conditional_score_drop": conditional_drop,
            })

            continue

        # The combined mask preserves the class.
        # A strong positive conditional drop suggests that the
        # neighbor supports the target and should not be absorbed
        # automatically into the exterior.
        if conditional_drop > min_score_drop:
            update_membership(
                memberships, neighbor_mask, 0, alpha
            )

            results.append({
                "exterior_id": exterior_id,
                "neighbor_id": neighbor_id,
                "status": "not_merged",
                "reason": "neighbor_supports_target_confidence",
                "base_outcome": base_outcome,
                "combined_outcome": combined_outcome,
                "conditional_score_drop": conditional_drop,
            })

            continue

        # The neighbor has sufficiently weak target-class support.
        # Merge it into the exterior region.
        update_membership(
            memberships, neighbor_mask, 1, alpha
        )

        old_exterior_id = exterior_id
        merged_id = lattice.merge(exterior_id, neighbor_id)

        # Continue growing from the new region.
        exterior_id = merged_id

        # Refresh region statistics used by later merge priorities.
        lattice.update_priority(memberships[..., 2])

        results.append({
            "exterior_id_before": old_exterior_id,
            "neighbor_id": neighbor_id,
            "merged_id": merged_id,
            "status": "merged",
            "reason": "weak_conditional_target_support",
            "base_outcome": base_outcome,
            "combined_outcome": combined_outcome,
            "conditional_score_drop": conditional_drop,
            "merged_area_pixels": int(np.count_nonzero(combined_mask))
        })

    return results




def shrink_interior(
    lattice,
    oracle,
    memberships,
    interior_id,
    entropy_tracker,
    samples_per_step=2,
    alpha=0.25,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng
    results = []

    if interior_id not in lattice.active_ids:
        return [{"status": "inactive_region", "region_id": interior_id}]

    node = lattice.nodes[interior_id]
    parent_mask = region_mask(lattice, interior_id)

    if not node.decompositions:
        return [{
            "interior_id": interior_id,
            "status": "needs_leaf_refinement",
        }]

    parent_outcome = oracle.evaluate(parent_mask)

    # A candidate must have some evidence of supporting the
    # original class before we interpret its children.
    if not parent_outcome["changed"]:
        update_membership(memberships, parent_mask, 2, alpha)
        lattice.update_priority(memberships[..., 2])
        return [{
            "interior_id": interior_id,
            "status": "parent_not_validated_as_interior",
            "outcome": parent_outcome,
        }]

    # STRUCTURAL CHANGE: restore the parent's children in the
    # active partition. The parent will no longer be active.
    child_ids = list(node.decompositions[-1])

    try:
        lattice.split(interior_id, decomposition=-1)
    except (ValueError, KeyError, RuntimeError) as exc:
        return [{
            "interior_id": interior_id,
            "status": "split_failed",
            "error": str(exc),
        }]

    # Keep only children that are now active and lie within
    # the former parent region.
    child_ids = [
        cid for cid in child_ids
        if cid in lattice.active_ids
        and lattice.nodes[cid].leaves.issubset(node.leaves)
    ]

    if not child_ids:
        return [{
            "interior_id": interior_id,
            "status": "split_created_no_active_children",
        }]

    # Prefer uncertain children, while still sampling stochastically.
    child_scores = []
    for cid in child_ids:
        mask = region_mask(lattice, cid)
        mu = memberships[mask].mean(axis=0)
        entropy = -np.sum(
            mu * np.log2(np.clip(mu, 1e-12, 1.0)),
            axis=-1,
        ).mean() / np.log2(3)

        outcome_entropy = entropy_tracker.priority_entropy(cid)
        boundary = float(mu[2])

        score = 0.45 * entropy + 0.35 * outcome_entropy + 0.20 * boundary
        child_scores.append(max(float(score), 1e-8))

    scores = np.asarray(child_scores, dtype=float)
    probabilities = scores / scores.sum()
    k = min(samples_per_step, len(child_ids))

    selected_indices = rng.choice(
        len(child_ids), size=k, replace=False, p=probabilities
    )

    for idx in selected_indices:
        child_id = child_ids[int(idx)]
        child_mask = region_mask(lattice, child_id)

        # Restore this child while keeping the rest of the parent
        # masked. Compare with masking the entire parent.
        perturb_mask = parent_mask & ~child_mask
        restored_outcome = oracle.evaluate(perturb_mask)

        changed_relative_to_parent = (
            restored_outcome["changed"] != parent_outcome["changed"]
        )
        entropy_tracker.record(child_id, changed_relative_to_parent)

        if not restored_outcome["changed"]:
            category = 0
            reason = "restoration_recovers_original_class"
        elif restored_outcome["score_drop"] < parent_outcome["score_drop"]:
            category = 0
            reason = "restoration_improves_target_score"
        else:
            category = 2
            reason = "child_effect_ambiguous"

        update_membership(memberships, child_mask, category, alpha)

        results.append({
            "interior_id": interior_id,
            "child_id": child_id,
            "status": "child_tested",
            "reason": reason,
            "parent_outcome": parent_outcome,
            "restored_outcome": restored_outcome,
        })

    lattice.update_priority(memberships[..., 2])
    return results



def discover_coarse_candidates(
    lattice,
    oracle,
    memberships,
    min_area_fraction=0.03,
    max_candidates=20,
    alpha=0.25,
):
    """
    Test large active regions individually.

    A changed prediction suggests interior/supporting evidence;
    an unchanged prediction suggests exterior/insensitive evidence.
    """
    image_area = lattice.base_labels.size

    candidates = sorted(
        [
            rid for rid in lattice.active_ids
            if lattice.nodes[rid].area / image_area
            >= min_area_fraction
        ],
        key=lambda rid: lattice.nodes[rid].area,
        reverse=True,
    )[:max_candidates]

    results = []

    for region_id in candidates:
        mask = region_mask(lattice, region_id)
        outcome = oracle.evaluate(mask)

        category = 0 if outcome["changed"] else 1

        update_membership(
            memberships, mask, category, alpha
        )

        results.append({
            "region_id": region_id,
            "area_fraction": lattice.nodes[region_id].area / image_area,
            "outcome": outcome,
            "candidate": (
                "interior" if category == 0 else "exterior"
            ),
        })

    return results



class OutcomeEntropyTracker:
    def __init__(self, min_observations=3):
        self.effects = {}
        self.min_observations = min_observations

    def record(self, region_id, effect_present):
        self.effects.setdefault(region_id, []).append(
            int(bool(effect_present))
        )

    def count(self, region_id):
        return len(self.effects.get(region_id, []))

    def entropy(self, region_id):
        outcomes = self.effects.get(region_id, [])

        if len(outcomes) < self.min_observations:
            return None

        q = float(np.mean(outcomes))

        if q <= 0.0 or q >= 1.0:
            return 0.0

        return float(
            -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))
        )

    def priority_entropy(self, region_id):
        """Exploration score: underobserved regions receive priority."""
        h = self.entropy(region_id)
        return 1.0 if h is None else h



def record_table_effects(table, tracker):
    """
    For rows differing in one region attribute, record whether adding
    that region changes the decision in the matched context.
    """
    X, y = table.arrays()

    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            differing = np.flatnonzero(X[i] != X[j])

            if len(differing) != 1:
                continue

            k = int(differing[0])
            region_id = table.attribute_ids[k]

            tracker.record(
                region_id,
                changes_decision=bool(y[i] != y[j]),
            )

def mean_region_membership(lattice, memberships, region_id):
    mask = region_mask(lattice, region_id)
    return memberships[mask].mean(axis=0)

def merge_and_test_boundary(
    lattice,
    oracle,
    memberships,
    entropy_tracker,
    max_merges=1,
    alpha=0.25,
    min_score_drop=0.05,
    lambda_boundary=0.25,
    min_merge_priority=0.05,
):
    """
    Merge adjacent boundary-like regions, then test the merged region.

    Membership channels:
        0 = interior
        1 = exterior
        2 = boundary

    score_drop = original target-class probability
                 - probability after masking the region
    """
    results = []

    for _ in range(max_merges):
        candidates = rank_boundary_merge_candidates(
            lattice=lattice,
            memberships=memberships,
            entropy_tracker=entropy_tracker,
            lambda_boundary=lambda_boundary,
        )

        # Keep only sufficiently boundary-like candidates.
        candidates = [
            c for c in candidates
            if c["priority"] >= min_merge_priority
        ]

        if not candidates:
            break

        merged_id = None
        selected_candidate = None

        # The ranking can contain stale pairs if an earlier merge
        # changed the graph, so try candidates until one succeeds.
        for candidate in candidates:
            u, v = candidate["u"], candidate["v"]

            if u not in lattice.active_ids or v not in lattice.active_ids:
                continue

            if (u, v) not in lattice.adjacency() and (
                v, u
            ) not in lattice.adjacency():
                continue

            merged_id = lattice.merge(u, v)
            selected_candidate = candidate
            break

        if merged_id is None:
            break

        merged_mask = region_mask(lattice, merged_id)
        outcome = oracle.evaluate(merged_mask)
        score_drop = float(outcome["score_drop"])
        changed = bool(outcome["changed"])

        # Use score magnitude and decision change together.
        if score_drop >= min_score_drop:
            category = 0
            reason = "target_class_support"
        elif score_drop <= -min_score_drop:
            category = 1
            reason = "masking_improves_target_class_score"
        elif not changed:
            category = 1
            reason = "prediction_preserved_with_small_score_change"
        else:
            category = 2
            reason = "decision_changed_but_score_evidence_is_weak"

        update_membership(
            memberships,
            merged_mask,
            category,
            alpha=alpha,
        )

        # Ensure future merge priorities use the latest boundary map.
        lattice.update_priority(memberships[..., 2])

        results.append({
            "merged_id": merged_id,
            "children": (selected_candidate["u"],
                         selected_candidate["v"]),
            "merge_priority": selected_candidate["priority"],
            "boundary_membership_before_merge":
                selected_candidate["boundary_membership"],
            "outcome_entropy_before_merge":
                selected_candidate["outcome_entropy"],
            "outcome": outcome,
            "assigned_category": category,
            "reason": reason,
        })

    return results

def split_priority(lattice, memberships, entropy_tracker, rid):
    mask = region_mask(lattice, rid)
    area_fraction = float(mask.mean())

    mu = memberships[mask].mean(axis=0)
    pixel_entropy = -np.sum(
        mu * np.log2(np.clip(mu, 1e-12, 1.0)),
        axis=-1,
    ).mean() / np.log2(3)
    boundary_membership = float(mu[2])
    outcome_entropy = entropy_tracker.priority_entropy(rid)

    # Favor large regions that remain uncertain or boundary-like.
    return (0.2 * pixel_entropy
    + 0.15 * outcome_entropy
    + 0.15 * boundary_membership
    + 0.5 * area_fraction * (
        0.4 * pixel_entropy
        + 0.3 * outcome_entropy
        + 0.3 * boundary_membership))

def adaptive_inference_step(
    lattice,
    oracle,
    memberships,
    entropy_tracker,
    max_boundary_merges=1,
    max_growth_candidates=2,
    max_shrink_candidates=2,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng

    log = {
        "boundary_merges": [],
        "exterior_growth": [],
        "interior_shrink": [],
    }

    # 1. Merge adjacent boundary-like regions.
    log["boundary_merges"] = merge_and_test_boundary(
        lattice=lattice,
        oracle=oracle,
        memberships=memberships,
        entropy_tracker=entropy_tracker,
        max_merges=max_boundary_merges,
    )

    # 2. Rank active regions for exterior growth and interior shrink.
    exterior_candidates = []
    interior_candidates = []

    for rid in list(lattice.active_ids):
        mu = mean_region_membership(
            lattice, memberships, rid
        )

        h_membership = membership_entropy(
            memberships,
            region_mask(lattice, rid),
        ) / np.log2(3)
        h_membership = float(
            np.clip(h_membership, 0.0, 1.0)
        )

        h_outcome = entropy_tracker.priority_entropy(rid)

        # Uncertainty prioritizes exploration within each category.
        priority = 0.5 * h_membership + 0.5 * h_outcome

        if mu[1] > mu[0] and mu[1] > mu[2]:
            exterior_candidates.append((priority, rid))

        if mu[0] > mu[1] and mu[0] > mu[2]:
            priority = split_priority(lattice, memberships, entropy_tracker, rid,)
            interior_candidates.append((priority, rid))

    exterior_candidates.sort(reverse=True)
    interior_candidates.sort(reverse=True)

    # 3. Grow exterior candidates.
    for _, rid in exterior_candidates[:max_growth_candidates]:
        if rid not in lattice.active_ids:
            continue

        results = grow_exterior(
            lattice=lattice,
            oracle=oracle,
            memberships=memberships,
            exterior_id=rid,
            entropy_tracker=entropy_tracker,
            count=3,
            rng=rng,
        )
        log["exterior_growth"].extend(results)

    # 4. Shrink interior candidates.
    for _, rid in interior_candidates[:max_shrink_candidates]:
        if rid not in lattice.active_ids:
            continue

        results = shrink_interior(
            lattice=lattice,
            oracle=oracle,
            memberships=memberships,
            interior_id=rid,
            entropy_tracker=entropy_tracker,
            samples_per_step=1,
            rng=rng,
        )
        log["interior_shrink"].extend(results)

    # Keep merge priorities synchronized with the latest memberships.
    lattice.update_priority(memberships[..., 2])

    return log

def region_boundary_stats(
    lattice,
    memberships,
    entropy_tracker,
    region_id,
    lambda_boundary=0.25,
):
    mask = region_mask(lattice, region_id)

    if not np.any(mask):
        return None

    boundary_membership = float(
        memberships[..., 2][mask].mean()
    )

    outcome_entropy = entropy_tracker.entropy(region_id)
    exploration_entropy = entropy_tracker.priority_entropy(region_id)

    merge_priority = boundary_membership * (
        lambda_boundary
        + (1.0 - lambda_boundary) * exploration_entropy
    )

    return {
        "region_id": region_id,
        "boundary_membership": boundary_membership,
        "outcome_entropy": outcome_entropy,
        "n_observations": entropy_tracker.count(region_id),
        "underexplored": outcome_entropy is None,
        "merge_priority": float(merge_priority),
    }
def rank_boundary_merge_candidates(
    lattice,
    memberships,
    entropy_tracker,
    lambda_boundary=0.25,
):
    candidates = []

    for u, v in lattice.adjacency():
        stats_u = region_boundary_stats(
            lattice, memberships, entropy_tracker, u,
            lambda_boundary,
        )
        stats_v = region_boundary_stats(
            lattice, memberships, entropy_tracker, v,
            lambda_boundary,
        )

        if stats_u is None or stats_v is None:
            continue

        pair_boundary = min(
            stats_u["boundary_membership"],
            stats_v["boundary_membership"],
        )

        pair_entropy = 0.5 * (
            entropy_tracker.priority_entropy(u)
            + entropy_tracker.priority_entropy(v)
        )

        priority = pair_boundary * (
            lambda_boundary
            + (1.0 - lambda_boundary) * pair_entropy
        )

        candidates.append({
            "u": u,
            "v": v,
            "priority": float(priority),
            "boundary_membership": pair_boundary,
            "outcome_entropy": pair_entropy,
        })

    return sorted(
        candidates,
        key=lambda item: item["priority"],
        reverse=True,
    )