"""Equivalence tests for active-set compaction of the recurrent forward pass.

Compaction shrinks the space the dense `Whh` is built over, from every candidate
position to the union any genome actually reached. At depth 7 that is the difference
between a 30.5 GB per-genome matrix and a negligible one.

The risk it carries is specific: the index remap must be **exact**. A wrong index
does not raise: it folds h->h edges onto the wrong nodes and returns numbers that
look entirely reasonable. So these tests assert exactness against a dense reference
rather than checking that results are merely close or merely finite.
"""

from __future__ import annotations

import numpy as np
import pytest

from emr_hyperneat.execution.compaction import (
    ActiveSet,
    compact_weights,
    find_active_positions,
    remap_sparse_hh,
)
from emr_hyperneat.emrhyperneat import (
    SparseHiddenConnections,
)

POP, N_IN, N_OUT, TOTAL = 4, 3, 2, 40


def make_masks(active_positions, pop=POP, total=TOTAL):
    """Population mask where the union of reached positions is exactly the input."""
    masks = np.zeros((pop, total), dtype=bool)
    for index, position in enumerate(active_positions):
        masks[index % pop, position] = True
    return masks


def make_sparse(from_idx, to_idx, weights=None, valid=None):
    n = len(from_idx)
    return SparseHiddenConnections(
        from_indices=np.array([from_idx], dtype=np.int32),
        to_indices=np.array([to_idx], dtype=np.int32),
        weights=np.array([weights if weights is not None else [1.0] * n],
                         dtype=np.float32),
        valid_mask=np.array([valid if valid is not None else [True] * n], dtype=bool),
        num_valid=np.array([n if valid is None else int(sum(valid))], dtype=np.int32),
    )


class TestFindActivePositions:
    def test_union_across_the_population(self):
        """Active means reached by *any* genome, not by all of them."""
        masks = np.zeros((3, 10), dtype=bool)
        masks[0, 2] = True
        masks[1, 7] = True
        masks[2, 2] = True
        active = find_active_positions(masks)
        assert active.indices.tolist() == [2, 7]
        assert active.num_active == 2

    def test_indices_are_sorted(self):
        """searchsorted correctness depends on this, so it is asserted, not assumed."""
        masks = make_masks([31, 2, 17, 8])
        active = find_active_positions(masks)
        assert active.indices.tolist() == sorted(active.indices.tolist())

    def test_nothing_active_is_representable(self):
        active = find_active_positions(np.zeros((3, 10), dtype=bool))
        assert active.num_active == 0
        assert active.indices.size == 0

    def test_everything_active_saves_nothing(self):
        active = find_active_positions(np.ones((3, 10), dtype=bool))
        assert active.num_active == 10
        assert active.reduction == pytest.approx(1.0)

    def test_reduction_reports_the_saving(self):
        active = find_active_positions(make_masks([5, 9], total=1000))
        assert active.reduction == pytest.approx(500.0)

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match="pop_size, total_positions"):
            find_active_positions(np.zeros(10, dtype=bool))


class TestCompactWeights:
    def test_gathers_the_position_axis_of_both_matrices(self):
        active = ActiveSet(indices=np.array([1, 3]), num_active=2, total_positions=TOTAL)
        W1 = np.arange(POP * N_IN * TOTAL, dtype=np.float32).reshape(POP, N_IN, TOTAL)
        W2 = np.arange(POP * N_OUT * TOTAL, dtype=np.float32).reshape(POP, N_OUT, TOTAL)
        W1_c, W2_c = compact_weights(W1, W2, active)
        assert W1_c.shape == (POP, N_IN, 2)
        assert W2_c.shape == (POP, N_OUT, 2)
        np.testing.assert_array_equal(np.asarray(W1_c), W1[:, :, [1, 3]])
        np.testing.assert_array_equal(np.asarray(W2_c), W2[:, :, [1, 3]])

    def test_rejects_positions_on_the_wrong_axis(self):
        """W2 is (pop, n_out, positions) on the host but transposed at evaluation.

        Gathering the output axis instead of the position axis is the mistake this
        guard exists for; it produces wrong numbers rather than an error.
        """
        active = ActiveSet(indices=np.array([1]), num_active=1, total_positions=TOTAL)
        transposed = np.zeros((POP, TOTAL, N_OUT), dtype=np.float32)
        with pytest.raises(ValueError, match="positions last"):
            compact_weights(np.zeros((POP, N_IN, TOTAL), dtype=np.float32),
                            transposed, active)


class TestRemapSparseHH:
    def test_active_indices_map_exactly(self):
        """The core correctness claim: a genuinely active index round-trips."""
        active = find_active_positions(make_masks([5, 12, 30]))
        sparse = make_sparse([5, 12, 30], [30, 5, 12])
        out = remap_sparse_hh(sparse, active)
        # active = [5, 12, 30] -> compact 0, 1, 2
        assert np.asarray(out.from_indices)[0].tolist() == [0, 1, 2]
        assert np.asarray(out.to_indices)[0].tolist() == [2, 0, 1]

    def test_edges_keep_their_endpoints(self):
        """An edge must connect the same two positions before and after remapping."""
        positions = [3, 11, 19, 27]
        active = find_active_positions(make_masks(positions))
        lookup = {p: i for i, p in enumerate(active.indices.tolist())}
        sparse = make_sparse([3, 19, 27], [27, 3, 11])
        out = remap_sparse_hh(sparse, active)
        for original, compact in zip([3, 19, 27], np.asarray(out.from_indices)[0]):
            assert compact == lookup[original]
        for original, compact in zip([27, 3, 11], np.asarray(out.to_indices)[0]):
            assert compact == lookup[original]

    def test_weights_and_validity_pass_through_untouched(self):
        active = find_active_positions(make_masks([2, 6]))
        sparse = make_sparse([2, 6], [6, 2], weights=[0.25, -0.75],
                             valid=[True, False])
        out = remap_sparse_hh(sparse, active)
        np.testing.assert_array_equal(np.asarray(out.weights), np.asarray(sparse.weights))
        np.testing.assert_array_equal(np.asarray(out.valid_mask),
                                      np.asarray(sparse.valid_mask))

    def test_invalid_entries_stay_in_range(self):
        """Padding may point anywhere; it must be clipped, and valid_mask zeroes it."""
        active = find_active_positions(make_masks([4, 9]))
        sparse = make_sparse([4, 9, 999], [9, 4, 999], valid=[True, True, False])
        out = remap_sparse_hh(sparse, active)
        assert np.asarray(out.from_indices).max() <= active.num_active - 1
        assert np.asarray(out.to_indices).max() <= active.num_active - 1
        assert np.asarray(out.from_indices).min() >= 0

    def test_empty_active_set_raises_rather_than_silently_degrading(self):
        empty = ActiveSet(indices=np.array([], dtype=int), num_active=0,
                          total_positions=TOTAL)
        with pytest.raises(ValueError, match="empty active set"):
            remap_sparse_hh(make_sparse([1], [1]), empty)


class TestDenseEquivalence:
    """The whole point: compaction must not change the computed Whh."""

    @staticmethod
    def build_whh(from_idx, to_idx, weights, valid, size):
        """Dense Whh exactly as the recurrent path builds it."""
        whh = np.zeros((size, size), dtype=np.float32)
        for f, t, w, v in zip(from_idx, to_idx, weights, valid):
            if v:
                whh[f, t] += w
        return whh

    @pytest.mark.parametrize("positions,edges", [
        ([5, 12, 30], [(5, 30, 0.5), (12, 5, -0.25), (30, 12, 1.5)]),
        ([0, 39], [(0, 39, 2.0), (39, 0, -2.0)]),
        ([7], [(7, 7, 0.75)]),  # self-loop
        ([2, 3, 4, 5], [(2, 5, 1.0), (3, 3, 1.0), (4, 2, 1.0), (5, 4, 1.0)]),
    ])
    def test_compact_whh_matches_dense_restricted_to_active(self, positions, edges):
        active = find_active_positions(make_masks(positions))
        lookup = {p: i for i, p in enumerate(active.indices.tolist())}

        from_idx = [e[0] for e in edges]
        to_idx = [e[1] for e in edges]
        weights = [e[2] for e in edges]
        valid = [True] * len(edges)

        dense = self.build_whh(from_idx, to_idx, weights, valid, TOTAL)
        sparse = make_sparse(from_idx, to_idx, weights, valid)
        out = remap_sparse_hh(sparse, active)
        compact = self.build_whh(
            np.asarray(out.from_indices)[0].tolist(),
            np.asarray(out.to_indices)[0].tolist(),
            weights, valid, active.num_active,
        )

        # The dense matrix restricted to active rows/cols must equal the compact one.
        restricted = dense[np.ix_(active.indices, active.indices)]
        np.testing.assert_array_equal(compact, restricted)
        # And nothing outside the active set carried any weight to begin with.
        assert dense.sum() == pytest.approx(restricted.sum())
        assert len(lookup) == active.num_active

    def test_a_deliberately_wrong_remap_is_caught(self):
        """Confirms the equivalence check has teeth.

        Collapsing every edge onto node 0 is the exact failure mode a bad remap
        produces, and it must not pass.
        """
        positions = [5, 12, 30]
        active = find_active_positions(make_masks(positions))
        edges = [(5, 30, 0.5), (12, 5, -0.25)]
        dense = self.build_whh([e[0] for e in edges], [e[1] for e in edges],
                               [e[2] for e in edges], [True] * 2, TOTAL)
        wrong = self.build_whh([0, 0], [0, 0], [e[2] for e in edges],
                               [True] * 2, active.num_active)
        restricted = dense[np.ix_(active.indices, active.indices)]
        assert not np.array_equal(wrong, restricted)


class TestEndToEndEquivalence:
    """The end-to-end test: a real recurrent generation, compacted vs not.

    The unit tests above prove the remap is exact on constructed inputs. This one
    checks the wiring end to end, by running the same seed twice (once normally,
    once with compaction neutralized so every position stays active) and requiring
    the same fitness to within 1e-6. Neutralizing rather than deleting keeps every other step
    (discovery, transfer, chunking, evaluation) on the same code path, so a
    difference can only come from compaction.
    """

    @staticmethod
    def _run_one_generation(monkeypatch, neutralize: bool):
        import numpy as np
        from conftest import XORProblem, create_config_with_recurrence
        from emr_hyperneat import EMRHyperNEAT
        from emr_hyperneat.execution import ExecutionMode
        from emr_hyperneat.execution import (
            compaction as compaction_module,
        )
        from emr_hyperneat.execution import (
            offload as offload_module,
        )

        if neutralize:
            def all_positions(masks):
                masks_np = np.asarray(masks)
                total = int(masks_np.shape[1])
                return compaction_module.ActiveSet(
                    indices=np.arange(total), num_active=total, total_positions=total
                )
            monkeypatch.setattr(offload_module, "find_active_positions", all_positions)

        config = create_config_with_recurrence(
            "hidden_only", max_depth=2, population_size=8, iteration_level=2
        )
        algorithm = EMRHyperNEAT()
        algorithm.execution_config.mode = ExecutionMode.DEPTH_OFFLOAD
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=7)
        _, metrics = algorithm.run_generation(state, problem)
        return float(metrics.best_fitness), float(metrics.mean_fitness)

    def test_compaction_does_not_change_fitness(self, monkeypatch):
        compacted = self._run_one_generation(monkeypatch, neutralize=False)
        uncompacted = self._run_one_generation(monkeypatch, neutralize=True)
        assert compacted == pytest.approx(uncompacted, abs=1e-6), (
            f"compaction changed the result: {compacted} vs {uncompacted}"
        )

    def test_compaction_actually_engaged(self, monkeypatch):
        """Guards against the equivalence above passing because nothing happened.

        At depth 2 the ceiling is 84 positions and XOR realizes a handful, so the
        active set must be a strict subset. If it were not, the test above would be
        comparing a path against itself.
        """
        from conftest import XORProblem, create_config_with_recurrence
        from emr_hyperneat import EMRHyperNEAT
        from emr_hyperneat.execution import ExecutionMode
        from emr_hyperneat.execution import (
            offload as offload_module,
        )

        seen = {}
        original = offload_module.find_active_positions

        def recording(masks):
            result = original(masks)
            seen["active"] = result.num_active
            seen["total"] = result.total_positions
            return result

        monkeypatch.setattr(offload_module, "find_active_positions", recording)

        config = create_config_with_recurrence(
            "hidden_only", max_depth=2, population_size=8, iteration_level=2
        )
        algorithm = EMRHyperNEAT()
        algorithm.execution_config.mode = ExecutionMode.DEPTH_OFFLOAD
        problem = XORProblem()
        state = algorithm.initialize(algorithm.create_config(config), problem, seed=7)
        algorithm.run_generation(state, problem)

        assert "active" in seen, "the recurrent offload path never ran"
        assert seen["active"] < seen["total"], (
            f"compaction saved nothing: {seen['active']} of {seen['total']} active"
        )
