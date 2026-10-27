
import numpy as np


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
            lattice, memberships, rid
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
    rng=None,
):
    exterior_mask = region_mask(lattice, exterior_id)
    base_outcome = oracle.evaluate(exterior_mask)

    # A region that disrupts the original decision is not currently
    # a valid exterior-growth seed.
    if base_outcome["changed"]:
        update_membership(
            memberships, exterior_mask, 2, alpha
        )
        lattice.update_priority(memberships[..., 2])

        return [{
            "exterior_id": exterior_id,
            "status": "seed_reclassified_as_boundary",
            "outcome": base_outcome,
        }]

    sampled = sample_neighbors(
        lattice, exterior_id, memberships, count, rng
    )
    results = []

    for neighbor_id in sampled:
        neighbor_mask = region_mask(lattice, neighbor_id)

        combined = oracle.evaluate(
            exterior_mask | neighbor_mask
        )

        # Does adding the neighbor change the result in this context?
        entropy_tracker.record(
            neighbor_id,
            combined["changed"] != base_outcome["changed"],
        )

        if not combined["changed"]:
            category = 1  # exterior evidence
            reason = "combined_mask_preserves_class"

        else:
            individual = oracle.evaluate(neighbor_mask)

            if individual["changed"]:
                category = 0
                reason = "neighbor_has_individual_supporting_evidence"
            else:
                category = 2
                reason = "context_dependent_effect"

        update_membership(
            memberships, neighbor_mask, category, alpha
        )

        results.append({
            "exterior_id": exterior_id,
            "neighbor_id": neighbor_id,
            "combined": combined,
            "reason": reason,
        })

    # Keep the lattice's boundary-priority statistics synchronized.
    lattice.update_priority(memberships[..., 2])

    return results



def shrink_interior(
    lattice,
    oracle,
    memberships,
    interior_id,
    entropy_tracker,
    samples_per_step=1,
    alpha=0.25,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng

    parent_mask = region_mask(lattice, interior_id)
    parent_outcome = oracle.evaluate(parent_mask)
    node = lattice.nodes[interior_id]

    if not node.decompositions:
        return []

    children = list(node.decompositions[-1])
    children = [
        child for child in children
        if lattice.nodes[child].leaves.issubset(node.leaves)
    ]

    if not children:
        return []

    # Without a changed decision from masking the parent, the assumed
    # interior status has not been established.
    if not parent_outcome["changed"]:
        update_membership(
            memberships, parent_mask, 2, alpha
        )
        lattice.update_priority(memberships[..., 2])

        return [{
            "interior_id": interior_id,
            "status": "parent_not_validated_as_interior",
            "outcome": parent_outcome,
        }]

    k = min(samples_per_step, len(children))
    selected = rng.choice(children, size=k, replace=False)
    results = []

    for child_id in selected:
        child_id = int(child_id)
        child_mask = region_mask(lattice, child_id)

        # Restore this child while leaving the rest of the parent masked.
        perturb_mask = parent_mask & ~child_mask
        restored_outcome = oracle.evaluate(perturb_mask)

        # Compare with the parent masked as a whole.
        entropy_tracker.record(
            child_id,
            restored_outcome["changed"] != parent_outcome["changed"],
        )

        if not restored_outcome["changed"]:
            category = 0
            reason = "restoration_recovers_original_class"
        else:
            category = 2
            reason = "restoration_does_not_recover_original_class"

        update_membership(
            memberships, child_mask, category, alpha
        )

        results.append({
            "interior_id": interior_id,
            "restored_child_id": child_id,
            "parent_outcome": parent_outcome,
            "restored_outcome": restored_outcome,
            "reason": reason,
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
            lattice, memberships, rid
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