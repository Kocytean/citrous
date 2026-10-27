
from itertools import combinations
import numpy as np


class PerturbationDecisionTable:
    """
    Each row describes one perturbation configuration.

    Attributes: binary indicators for candidate regions being perturbed.
    Decision: 1 if the predicted class changed, otherwise 0.
    """

    def __init__(self, attribute_ids):
        self.attribute_ids = tuple(sorted(set(attribute_ids)))
        self.rows = []

    def add(self, perturbed_ids, outcome, context=None):
        perturbed_ids = set(perturbed_ids)

        row = {
            "attributes": {
                rid: int(rid in perturbed_ids)
                for rid in self.attribute_ids
            },
            "decision": int(outcome["changed"]),
            "score_drop": float(outcome["score_drop"]),
            "context": context,
            "prediction": int(outcome["prediction"]),
        }
        self.rows.append(row)

    def arrays(self):
        if not self.rows:
            return (
                np.empty((0, len(self.attribute_ids)), dtype=np.int8),
                np.empty((0,), dtype=np.int8),
            )

        X = np.asarray([
            [row["attributes"][rid] for rid in self.attribute_ids]
            for row in self.rows
        ], dtype=np.int8)

        y = np.asarray(
            [row["decision"] for row in self.rows],
            dtype=np.int8,
        )

        return X, y

    def contradictory_rows(self):
        """Configurations observed with conflicting decisions."""
        X, y = self.arrays()
        groups = {}

        for i, vector in enumerate(X):
            key = tuple(vector.tolist())
            groups.setdefault(key, set()).add(int(y[i]))

        return [
            key for key, decisions in groups.items()
            if len(decisions) > 1
        ]

    def find_reducts(self, max_attributes=16):
        """
        Find all minimal attribute subsets that distinguish every
        distinguishable pair of rows with different decisions.

        This is an exact minimal hitting-set search. Exponential in
        attribute count; use a small local table.
        """
        X, y = self.arrays()
        n_attributes = X.shape[1]

        if len(y) < 2:
            return {
                "reducts": [],
                "contradictory_configurations": self.contradictory_rows(),
                "note": "Need at least two observations.",
            }

        if n_attributes > max_attributes:
            raise ValueError(
                f"{n_attributes} attributes exceed the exact-search "
                f"limit ({max_attributes}). Reduce the local candidate set."
            )

        # A pair of opposite decisions must differ on at least one
        # selected attribute to be distinguished by a reduct.
        discernibility = set()
        impossible_pairs = []

        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if y[i] == y[j]:
                    continue

                differing = frozenset(
                    np.flatnonzero(X[i] != X[j]).tolist()
                )

                if not differing:
                    impossible_pairs.append((i, j))
                else:
                    discernibility.add(differing)

        if not discernibility:
            return {
                "reducts": [],
                "contradictory_configurations": self.contradictory_rows(),
                "impossible_pairs": impossible_pairs,
                "note": "No distinguishable opposite-decision pairs.",
            }

        # Search subsets in increasing size; retain only minimal reducts.
        constraints = list(discernibility)
        reducts = []

        for size in range(1, n_attributes + 1):
            for subset_tuple in combinations(range(n_attributes), size):
                subset = frozenset(subset_tuple)

                # A subset is a reduct candidate if it hits every
                # discernibility set.
                if not all(subset & constraint for constraint in constraints):
                    continue

                # Minimality: no previously found reduct is a subset.
                if any(r.issubset(subset) for r in reducts):
                    continue

                reducts.append(subset)

        readable_reducts = [
            tuple(self.attribute_ids[i] for i in sorted(r))
            for r in reducts
        ]

        support = {rid: 0.0 for rid in self.attribute_ids}

        if reducts:
            for rid in self.attribute_ids:
                index = self.attribute_ids.index(rid)
                support[rid] = float(
                    np.mean([index in reduct for reduct in reducts])
                )

        return {
            "reducts": readable_reducts,
            "support": support,
            "contradictory_configurations": self.contradictory_rows(),
            "impossible_pairs": impossible_pairs,
        }
