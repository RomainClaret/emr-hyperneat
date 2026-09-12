"""Tests for the EMR execution-mode configuration and chunk arithmetic.

The behaviour these lock down is mostly about what must *not* change: a config
with no `execution` block has to keep producing the standard path, because that
is what the published results and the golden tests were produced with.

The chunking tests pin the arithmetic that decides memory feasibility at depth
10+. They are exact-value tests rather than range checks -- an off-by-a-factor
here does not fail loudly, it just silently OOMs or wastes an order of magnitude
of memory.
"""

from __future__ import annotations

import pytest

from emr_hyperneat.execution import (  # noqa: E501
    ExecutionConfig,
    ExecutionMode,
    parse_execution_config,
    resolve_mode,
)
from emr_hyperneat.execution.chunking import (  # noqa: E501
    MIN_POPULATION_CHUNK,
    MIN_POSITION_CHUNK,
    optimal_position_chunk_size,
    population_chunk_size,
    recurrent_chunk_sizes,
    cppn_query_carry_bytes,
    population_chunk_for_query,
    LEGACY_CPPN_QUERY_OVERHEAD,
)


class TestExecutionModeDefaults:
    """The default has to be the pre-unification standard path, or goldens move."""

    def test_default_mode_is_standard(self):
        assert ExecutionConfig().mode is ExecutionMode.STANDARD

    def test_empty_config_yields_standard(self):
        assert parse_execution_config({}).mode is ExecutionMode.STANDARD

    def test_unrelated_keys_do_not_change_mode(self):
        config = parse_execution_config({"max_depth": 12, "variance_threshold": 0.03})
        assert config.mode is ExecutionMode.STANDARD

    def test_explicit_standard_is_not_overridden_by_depth(self):
        config = ExecutionConfig(mode=ExecutionMode.STANDARD)
        assert resolve_mode(config, max_depth=13) is ExecutionMode.STANDARD


class TestModeResolution:
    """AUTO is the only mode that consults depth."""

    @pytest.mark.parametrize(
        "depth,expected",
        [
            (1, ExecutionMode.STANDARD),
            (7, ExecutionMode.STANDARD),
            (8, ExecutionMode.STREAMING),
            (9, ExecutionMode.DEPTH_OFFLOAD),
            (13, ExecutionMode.DEPTH_OFFLOAD),
        ],
    )
    def test_auto_resolves_by_depth(self, depth, expected):
        assert resolve_mode(ExecutionConfig(mode=ExecutionMode.AUTO), depth) is expected

    def test_offload_threshold_wins_when_both_are_crossed(self):
        # Both thresholds are crossed at depth 9; the more conservative one applies.
        config = ExecutionConfig(
            mode=ExecutionMode.AUTO, streaming_depth_threshold=2, depth_threshold=3
        )
        assert resolve_mode(config, 9) is ExecutionMode.DEPTH_OFFLOAD

    def test_resolve_never_returns_auto(self):
        for depth in range(1, 15):
            assert resolve_mode(ExecutionConfig(mode=ExecutionMode.AUTO), depth) \
                is not ExecutionMode.AUTO

    def test_method_and_function_agree(self):
        config = ExecutionConfig(mode=ExecutionMode.AUTO)
        assert config.resolve(11) is resolve_mode(config, 11)


class TestLegacyStreamingFlag:
    """`enable_streaming` stays a valid spelling of `mode: streaming`."""

    def test_legacy_flag_selects_streaming(self):
        assert parse_execution_config(
            {"enable_streaming": True}
        ).mode is ExecutionMode.STREAMING

    def test_legacy_flag_false_stays_standard(self):
        assert parse_execution_config(
            {"enable_streaming": False}
        ).mode is ExecutionMode.STANDARD

    def test_explicit_mode_overrides_legacy_flag(self):
        config = parse_execution_config(
            {"enable_streaming": True, "execution": {"mode": "standard"}}
        )
        assert config.mode is ExecutionMode.STANDARD


class TestConfigParsing:
    def test_every_knob_is_reachable_from_config(self):
        """A knob that cannot be set from YAML is a knob that does not exist."""
        block = {
            "mode": "auto",
            "depth_threshold": 5,
            "streaming_depth_threshold": 4,
            "cppn_pos_chunk_size": 1234,
            "target_shard_gb": 0.25,
            "max_lookahead_positions": 99,
            "truncate_variance_lookahead": True,
            "eval_chunk_levels": 7,
            "eval_chunk_positions": 555,
            "target_eval_shard_gb": 2.5,
            "recurrent_target_memory_gb": 3.5,
            "target_cpu_gb": 8.0,
            "max_population_chunk": 64,
            "use_cpu_grid": True,
            "use_memmap": False,
            "memmap_path": "/tmp/custom_memmap",
            "enable_disk_offload": True,
            "disk_offload_path": "/tmp/custom_offload",
            "verbose": True,
        }
        config = parse_execution_config({"execution": block})
        for key, value in block.items():
            if key == "mode":
                continue
            assert getattr(config, key) == value, f"{key} did not round-trip"

    def test_mode_is_case_insensitive(self):
        assert parse_execution_config(
            {"execution": {"mode": "STREAMING"}}
        ).mode is ExecutionMode.STREAMING

    def test_unknown_mode_is_rejected_with_the_valid_list(self):
        with pytest.raises(ValueError, match="Unknown execution mode"):
            parse_execution_config({"execution": {"mode": "turbo"}})

    def test_non_mapping_execution_block_is_rejected(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_execution_config({"execution": "streaming"})

    def test_variance_truncation_is_off_by_default(self):
        """It changes the discovered substrate, so it may not be implicit."""
        assert parse_execution_config({}).truncate_variance_lookahead is False
        assert parse_execution_config(
            {"execution": {"mode": "depth_offload"}}
        ).truncate_variance_lookahead is False


class TestPositionChunking:
    """The carry-corrected model. Exact values changed deliberately:
    the old formula (`LEGACY_CPPN_QUERY_OVERHEAD`) modeled 2 floats per
    (genome, position) where the measured forward carry is 2 x max_nodes, a 50x
    undercount that sized the 62,500-position shard behind the recorded 25.0 GB
    depth-7 OOM."""

    def test_scales_inversely_with_population(self):
        big = optimal_position_chunk_size(100, 3, 1.0, 10_000_000, max_nodes=50)
        small = optimal_position_chunk_size(1000, 3, 1.0, 10_000_000, max_nodes=50)
        # Inverse-proportional up to the integer floor: 1e9/41200 = 24271.8
        # and 1e9/412000 = 2427.18.
        assert abs(big - 10 * small) <= 10

    def test_respects_the_configured_cap(self):
        assert optimal_position_chunk_size(1, 1, 100.0, 5000, max_nodes=50) == 5000

    def test_never_below_the_floor(self):
        assert optimal_position_chunk_size(
            1_000_000, 784, 0.001, 10_000_000, max_nodes=50
        ) == MIN_POSITION_CHUNK

    def test_exact_value(self):
        # 100 * 4 * (2*50 + 3) = 41,200 bytes per position; 1e9 / 41200 = 24271
        assert optimal_position_chunk_size(100, 3, 1.0, 10_000_000, max_nodes=50) == 24_271

    def test_shard_fits_the_target_it_claims(self):
        # The property the old formula violated: the carry of one emitted shard
        # must actually fit inside target_gb.
        chunk = optimal_position_chunk_size(1000, 1, 0.5, 100_000, max_nodes=50)
        assert cppn_query_carry_bytes(1000, chunk, 50) <= 0.5e9


class TestCarryByteModel:
    """Pinned against two recorded byte-exact depth-7 allocations."""

    OBSERVED_STREAMING = 6_555_023_048   # level-6 slab, 16,384 positions
    OBSERVED_OFFLOAD = 25_001_623_112    # 62,500-position W2 shard

    def test_matches_streaming_level6_allocation(self):
        predicted = cppn_query_carry_bytes(1000, 16_384, 50)
        assert predicted == 1000 * 16_384 * 100 * 4
        assert abs(self.OBSERVED_STREAMING - predicted) / predicted < 0.001

    def test_matches_offload_w2_allocation(self):
        predicted = cppn_query_carry_bytes(1000, 62_500, 50)
        assert predicted == 25_000_000_000
        assert abs(self.OBSERVED_OFFLOAD - predicted) / predicted < 0.001

    def test_legacy_model_undercounted_50x(self):
        # What the old sizing charged per (genome, position) for one source,
        # against the measured carry. The quotient is the recorded defect.
        legacy_bytes = 1 * 4 * LEGACY_CPPN_QUERY_OVERHEAD
        true_bytes = cppn_query_carry_bytes(1, 1, 50)
        assert true_bytes // legacy_bytes == 50


class TestAutoPopulationGate:
    """The -1 sentinel's new semantics: chunk only past the budget."""

    def test_fitting_configurations_are_untouched(self):
        # d1-d6 protocol shapes all fit 0.5 GB: the gate must return pop_size
        # unchanged so measured defaults stay byte-identical.
        for largest_level in (4, 16, 64, 256, 1024):  # d<=5 level slabs
            assert population_chunk_for_query(1000, largest_level, 50, 0.5) == 1000

    def test_depth7_level_slab_chunks_to_fit(self):
        # Level L holds 4^(L+1) positions, so depth 7's largest slab (level 7)
        # is 65,536; its carry at pop 1000 is 26.2 GB against a 0.5 GB budget.
        chunk = population_chunk_for_query(1000, 65_536, 50, 0.5)
        assert 1 <= chunk < 1000
        assert cppn_query_carry_bytes(chunk, 65_536, 50) <= 0.5e9
        assert chunk == 19  # 0.5e9 // (65_536 * 100 * 4)

    def test_degenerate_inputs_fall_through(self):
        assert population_chunk_for_query(1000, 0, 50, 0.5) == 1000
        assert population_chunk_for_query(0, 100, 50, 0.5) == 1


class TestPopulationChunking:
    @pytest.mark.parametrize(
        "positions,target_gb,expected",
        [
            (5_600_000, 16.0, 151),
            (5_600_000, 32.0, 302),
            (22_400_000, 16.0, 37),
            (22_400_000, 32.0, 75),
        ],
    )
    def test_documented_depth_10_and_11_cases(self, positions, target_gb, expected):
        """Pins the 10% safety margin, which the predecessor's docstring omitted."""
        assert population_chunk_size(1000, positions, target_gb) == expected

    def test_explicit_override_wins(self):
        assert population_chunk_size(1000, 22_400_000, 32.0, max_population_chunk=7) == 7

    def test_override_is_still_clamped_to_population(self):
        assert population_chunk_size(50, 340, 32.0, max_population_chunk=10_000) == 50

    def test_never_exceeds_population(self):
        assert population_chunk_size(20, 340, 1024.0) == 20

    def test_never_below_the_floor(self):
        assert population_chunk_size(
            1000, 500_000_000, 0.5
        ) == MIN_POPULATION_CHUNK

    def test_wider_io_shrinks_the_chunk(self):
        narrow = population_chunk_size(1000, 1_000_000, 32.0, num_inputs=3)
        wide = population_chunk_size(1000, 1_000_000, 32.0, num_inputs=784)
        assert wide < narrow


class TestRecurrentChunking:
    def test_chunks_both_axes(self):
        """Chunking population alone does not bound scan memory; both axes must move."""
        pop_chunk, sample_chunk = recurrent_chunk_sizes(
            1000, 87_380, 500, activate_time=8, target_memory_gb=2.0
        )
        assert pop_chunk < 1000
        assert sample_chunk < 500

    def test_longer_activation_shrinks_chunks(self):
        short = recurrent_chunk_sizes(1000, 87_380, 500, activate_time=1)
        long = recurrent_chunk_sizes(1000, 87_380, 500, activate_time=16)
        assert long[0] < short[0]

    def test_clamped_by_the_actual_sizes(self):
        pop_chunk, sample_chunk = recurrent_chunk_sizes(
            12, 340, 4, activate_time=1, target_memory_gb=64.0
        )
        assert pop_chunk == 12
        assert sample_chunk == 10  # the floor, since n_samples is below it

    def test_the_ooming_case_from_the_docstring(self):
        """19 x 1000 x 500 x 87380 x 4 bytes is 331 GB dense; chunking is the fix."""
        pop_chunk, sample_chunk = recurrent_chunk_sizes(
            1000, 87_380, 500, activate_time=19, target_memory_gb=2.0
        )
        resident_gb = pop_chunk * sample_chunk * 87_380 * 19 * 4 / 1e9
        assert resident_gb <= 2.0


class TestMemoryAwareResolution:
    """AUTO's GPU escalation: depth thresholds alone routed
    d7/pop1000 to STANDARD on 11 GB, which OOMed 0/10. Both anchors here are
    measured GPU cells."""

    GPU_USABLE = int(9.35e9)  # 11 GB card at XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

    def _cfg(self):
        # AUTO is opt-in: the dataclass default mode is STANDARD (documented as
        # "defaults reproduce the pre-unification behaviour"), so the gate is
        # only reachable through an explicit mode: auto.
        return parse_execution_config({"execution": {"mode": "auto"}})

    def test_d7_pop1000_escalates_to_offload(self):
        from emr_hyperneat.execution import (
            ExecutionMode, resolve_mode,
        )
        mode = resolve_mode(self._cfg(), 7, population=1000,
                            device_bytes=self.GPU_USABLE)
        assert mode is ExecutionMode.DEPTH_OFFLOAD

    def test_d6_pop1000_stays_standard(self):
        # The measured d6 GPU cell ran STANDARD clean (3.35 s/gen, n=10).
        from emr_hyperneat.execution import (
            ExecutionMode, resolve_mode,
        )
        mode = resolve_mode(self._cfg(), 6, population=1000,
                            device_bytes=self.GPU_USABLE)
        assert mode is ExecutionMode.STANDARD

    def test_without_sizing_resolution_is_unchanged(self):
        # CPU runs and legacy callers pass no sizing: pure threshold logic.
        from emr_hyperneat.execution import (
            ExecutionMode, resolve_mode,
        )
        assert resolve_mode(self._cfg(), 7) is ExecutionMode.STANDARD
        # AUTO's static ordering: streaming at 8, offload at 9.
        assert resolve_mode(self._cfg(), 8) is ExecutionMode.STREAMING
        assert resolve_mode(self._cfg(), 9) is ExecutionMode.DEPTH_OFFLOAD

    def test_explicit_mode_is_never_overridden(self):
        from emr_hyperneat.execution import (
            ExecutionMode, resolve_mode,
        )
        forced = parse_execution_config({"execution": {"mode": "standard"}})
        mode = resolve_mode(forced, 7, population=1000,
                            device_bytes=self.GPU_USABLE)
        assert mode is ExecutionMode.STANDARD

    def test_floor_model_matches_the_recorded_allocation(self):
        # The 1.398 GB protocol-d7 failure is byte-exact the 4-float query
        # output family: pop x positions x 16 B.
        from emr_hyperneat.execution.chunking import (
            standard_mode_floor_bytes,
        )
        positions = (4 ** 9 - 4) // 3
        query_family = 1000 * positions * 16
        assert query_family == 1_398_080_000
        assert standard_mode_floor_bytes(1000, positions) > query_family


class TestChunkProductBudget:
    """Round-4 fix: the offload path sizes position chunks and the population
    gate against the SAME shard target -- independently, in round 3, so the
    realized per-call carry ran 53x UNDER budget and auto-mode d7 cost 1.97x
    the hand-tuned cell. The gate now budgets
    against the realized slab (min(largest level, largest position chunk))."""

    MN = 50
    BUDGET = 0.5  # target_shard_gb default

    def test_realized_carry_lands_on_budget_at_d7(self):
        from emr_hyperneat.execution.chunking import (
            cppn_query_carry_bytes, optimal_position_chunk_size,
            population_chunk_for_query,
        )
        largest_level = 4 ** 8  # depth 7's level-7 slab
        pos_w1 = optimal_position_chunk_size(1000, 3, self.BUDGET, 100_000, self.MN)
        pos_w2 = optimal_position_chunk_size(1000, 1, self.BUDGET, 100_000, self.MN)
        realized_slab = min(largest_level, max(pos_w1, pos_w2))
        pop_chunk = population_chunk_for_query(1000, realized_slab, self.MN, self.BUDGET)
        carry = cppn_query_carry_bytes(pop_chunk, realized_slab, self.MN)
        budget_bytes = self.BUDGET * 1e9
        # Round 3's independent budgeting realized ~9.4 MB here (53x under).
        # The product budget must land within 2x of the target, either side.
        assert budget_bytes / 2 <= carry <= budget_bytes * 2, (
            f"realized carry {carry/1e6:.1f} MB vs budget {budget_bytes/1e6:.0f} MB")
        # And the launch count collapses: full population in one pop chunk here.
        assert pop_chunk == 1000

    def test_round3_independent_budgeting_was_53x_under(self):
        # Pin the defect being fixed, so the fix's magnitude stays on record.
        from emr_hyperneat.execution.chunking import (
            cppn_query_carry_bytes, optimal_position_chunk_size,
            population_chunk_for_query,
        )
        largest_level = 4 ** 8
        pos_w2 = optimal_position_chunk_size(1000, 1, self.BUDGET, 100_000, self.MN)
        old_pop_chunk = population_chunk_for_query(  # old slab: the level
            1000, largest_level, self.MN, self.BUDGET)
        old_carry = cppn_query_carry_bytes(old_pop_chunk, pos_w2, self.MN)
        assert (self.BUDGET * 1e9) / old_carry > 40  # ~53x, allow slack

    def test_fitting_configurations_still_uncapped(self):
        # d1-d6 slabs: position chunk covers the whole level, pop gate returns
        # the full population -- byte-identical values, no new chunking.
        from emr_hyperneat.execution.chunking import (
            optimal_position_chunk_size, population_chunk_for_query,
        )
        for level in (4, 16, 64, 256, 1024):
            pos = optimal_position_chunk_size(1000, 1, self.BUDGET, 100_000, self.MN)
            slab = min(level, pos)
            assert population_chunk_for_query(1000, slab, self.MN, self.BUDGET) == 1000
