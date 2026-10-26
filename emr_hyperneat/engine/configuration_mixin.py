"""Configuration parsing and its checks: the recurrence, dynamic-functions and neuromodulation sections,
the checks across features, and the recurrence and forward-mode accessors."""
import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.emrhyperneat_base import MultiGPUStrategy
from emr_hyperneat.engine.config import (
    EMRConfig,
    ForwardPassMode,
    HH_CACHE_TRIGGERS,
    NETWORK_MODELS,
    SETTLING_RULES,
    _refuse_unknown_keys,
    _validated_hh_discovery_mode,
    check_recurrence_switch,
    get_recurrence_preset,
    resolve_recurrence,
)
from emr_hyperneat.engine.discovery import _KEEP_CEILING, _sparse_cap_lock
from emr_hyperneat.engine.functions import (
    ACTIVATION_FUNCTIONS,
    ACTIVATION_INTERPRETATIONS,
    ACTIVATION_LIST,
    AGGREGATION_FUNCTIONS,
    AGGREGATION_INTERPRETATIONS,
    AGGREGATION_LIST,
    INDEX_DECODES,
    MODULAR_ACTIVATION_METHODS,
    MODULAR_AGGREGATION_METHODS,
    MODULAR_SCALING_METHODS,
    MODULAR_SPARSITY_METHODS,
    PALETTE_CONFIGS,
    PALETTE_EVOLUTION_KEYS,
    RANDOM_SCOPES,
    _parse_hmr_reads,
    parse_function_palette,
    parse_live_strategy,
    parse_palette_evolution,
)
from emr_hyperneat.engine.leo import LEO_CHANNEL
from emr_hyperneat.engine.multitask import (
    GENERALIST_BONUS_TYPES,
    MULTITASK_AGGREGATIONS,
    MULTITASK_FITNESS_MODES,
    MultiTaskConfig,
    ORTHOGONALITY_METRICS,
    TWO_MODULE_MODES,
    _refuse_per_node_in_multitask,
    check_branch_gating_and_penalty,
    refuse_anti_branch_gating,
)
from emr_hyperneat.engine.neuromodulation import (
    MODULATION_MODES,
    MODULATORY_OPTIONS,
    NEUROMODULATION_NODE_RULES,
    NEUROMODULATION_PRESETS,
    NEUROMODULATION_SOURCES,
    NeuromodulationConfig,
    _translate_hmr_neuromodulation,
    check_modulatory_options,
    get_neuromodulation_config,
    get_nt_for_task,
    modulatory_cppn_reads,
    neuromodulation_cppn_channels,
)
from emr_hyperneat.engine.settling import HH_ACTIVATION_MODES
from emr_hyperneat.engine.substrate_build import WEIGHT_THRESHOLD
from emr_hyperneat.execution import ExecutionMode


class ConfigurationMixin:
    """EMRHyperNEAT's configuration parsing and checks."""

    def _check_feature_combinations(self) -> None:
        """Refuse configurations one evaluation path would silently drop part of.

        Raises:
            ValueError: multi-task evaluation with dynamic functions or aggregation (it applies
                one activation per task and sum aggregation), multi-task two-module
                evaluation with recurrence (it is feed-forward), or the depth-offload execution
                mode or the EVAL_ONLY_PARALLEL strategy with dynamic functions, aggregation or
                neuromodulation (both evaluate tanh nodes with sum aggregation and no
                modulation).
        """
        self._check_function_palettes()
        recurrent = (self.extended_config is not None
                     and self.extended_config.allow_hidden_to_hidden)
        self._check_hmr_reads(recurrent)
        # Modulatory neurons reading a CPPN output at the input queries (modulatory_cppn_read
        # 'input_mean'); the self read takes the self-connection query, which streaming runs too
        modulatory_input_read = (self.neuromod_config is not None
                                 and bool(getattr(self, '_modulatory_cppn_reads', ()))
                                 and self.neuromod_config.modulatory_cppn_read == 'input_mean')
        if self.neuromod_config is not None and (
            self.neuromod_config.gate_source == 'cppn' or modulatory_input_read
        ):
            # Neuromodulation from dedicated CPPN outputs reads the raw queries of the standard path
            if self.execution_config.resolve(self.max_depth) is not ExecutionMode.STANDARD:
                raise ValueError(
                    "neuromodulation from dedicated CPPN outputs read at the input queries "
                    "(gate_source 'cppn', or modulatory neurons with modulatory_cppn_read "
                    "'input_mean', the CPPN source's default) reads the raw CPPN queries of the "
                    "standard execution path; set execution.mode standard, or for modulatory "
                    "neurons modulatory_cppn_read 'self'"
                )
            if self.neuromod_config.mode == 'context_gating':
                if recurrent:
                    raise ValueError(
                        "context gating from a CPPN output gates each case's connections in a "
                        "feed-forward pass (HMR's Level 2) and would ignore the hidden-to-hidden "
                        "connections"
                    )
                if self.agg_mode != 'disabled':
                    raise ValueError(
                        "context gating from a CPPN output computes each case's input drive as a "
                        "weighted sum and would ignore aggregation.mode"
                    )
        if self.neuromod_config is not None:
            rule = self.neuromod_config.node_rule
            if rule not in NEUROMODULATION_NODE_RULES:
                raise ValueError(
                    f"neuromodulation.node_rule must be one of {NEUROMODULATION_NODE_RULES}, got "
                    f"{rule!r}"
                )
            if (rule != 'paper' and self.multitask_config is not None
                    and getattr(self.multitask_config, 'two_module_mode', 'none') != 'none'):
                raise ValueError(
                    f"neuromodulation.node_rule {rule!r} is not applied by "
                    "multitask.two_module_mode, "
                    f"whose modules compute the paper's rule"
                )
            if self.multitask_config is not None:
                refuse_anti_branch_gating(
                    rule, getattr(self.multitask_config, 'branch_gating_mode', 'none')
                )
        if self._settling_rule == 'hmr':
            if not recurrent:
                raise ValueError(
                    "recurrence.settling_rule 'hmr' needs hidden-to-hidden recurrence; this "
                    "configuration is feed-forward, where both rules compute the same network"
                )
            if self.neuromod_config is not None and self.neuromod_config.mode != 'disabled':
                # HMR FULL's additive rule: what it cannot apply raises
                self._check_additive_settling()
        dropped = self._features_a_tanh_sum_path_drops()
        # The multi-task evaluation applies neuromodulation, and HMR FULL's multi-task rule under
        # settling_rule 'hmr'
        per_node = [feature for feature in dropped
                    if feature not in ("neuromodulation", "recurrence.settling_rule='hmr'")]
        multitask = (self.multitask_config is not None
                     and getattr(self.multitask_config, 'enabled', False))
        # The configured hidden activation applies in mode 'global' only, and the output activation
        # everywhere except the multi-task evaluation, which applies tanh and sigmoid; a setting it
        # would drop raises. The plain evaluation (no per-node function, aggregation or
        # neuromodulation) applies the output activation, as HMR's dynamic-functions variant did:
        # see below.
        if self.dynamic_functions_mode != 'global' and self.df_hidden_activation != 'tanh':
            raise ValueError(
                f"hidden_activation {self.df_hidden_activation!r} (dynamic_functions or substrate) "
                "applies in dynamic_functions.mode 'global' only; mode "
                f"{self.dynamic_functions_mode!r} would ignore it. Set mode 'global', or drop the "
                "key.")
        modulated = self.neuromod_config is not None and self.neuromod_config.mode != 'disabled'
        modular_aggregation = (self.df_modular_config or {}).get('aggregation_method', 'none')
        if modular_aggregation != 'none' and self.agg_mode != 'modular':
            # aggregation.mode 'modular' reads it, whatever the dynamic-functions mode (0.1.0
            # read it nowhere)
            raise ValueError(
                f"dynamic_functions.modular.aggregation_method {modular_aggregation!r} is read by "
                "aggregation.mode 'modular' only; this configuration has aggregation.mode "
                f"{self.agg_mode!r}, where it would be ignored. Set aggregation.mode 'modular', or "
                "drop the key.")
        no_aggregation = self.agg_mode == 'disabled' or (
            self.agg_mode == 'modular' and self.df_modular_config['aggregation_method'] == 'none')
        plain = self.dynamic_functions_mode == 'disabled' and no_aggregation and not modulated
        # HMR's sparsity mechanisms exist for feed-forward evaluation only (its dynamic-functions
        # variant has no recurrence), and the neuromodulated evaluation does not apply them.
        if self._df_sparsity_method != 'none' and (recurrent or modulated or multitask):
            raise ValueError(
                f"the {self._df_sparsity_method!r} sparsity of dynamic_functions "
                f"({self.dynamic_functions_mode} mode) applies to the feed-forward evaluation "
                "only; this configuration also has "
                + ", ".join(
                    name
                    for name, on in (
                        ('hidden-to-hidden recurrence', recurrent),
                        ('neuromodulation', modulated),
                        ('multi-task evaluation', multitask),
                    )
                    if on
                )
            )
        if (plain and not multitask and self.agg_mode == 'disabled'
                and self.df_output_activation != 'sigmoid'):
            # HMR's dynamic-functions variant applied the configured output activation in plain
            # mode. The plain evaluation equals mode 'global' with a tanh hidden activation bit for
            # bit (tests/test_hmr_names.py), so the run takes that path with the configured output
            # activation; dynamic_functions_mode then reads 'global'.
            self.dynamic_functions_mode = 'global'
            self.df_hidden_activation = 'tanh'
            self._hidden_act_fn = jnp.tanh
            self._plain_output_activation = True
        elif (plain or multitask) and self.df_output_activation != 'sigmoid':
            raise ValueError(
                f"output_activation {self.df_output_activation!r} (dynamic_functions or substrate) "
                f"would be ignored: the {'multi-task' if multitask else 'plain'} evaluation "
                "applies a sigmoid output. Set dynamic_functions.mode 'global' with "
                "hidden_activation 'tanh' to use another output activation.")
        if multitask:
            _refuse_per_node_in_multitask(per_node)
        if (multitask and recurrent
                and getattr(self.multitask_config, 'two_module_mode', 'none') != 'none'):
            raise ValueError(
                "multitask.two_module_mode evaluates feed-forward only and would ignore the "
                "hidden-to-hidden connections this configuration discovers"
            )
        if self.weight_threshold != WEIGHT_THRESHOLD and (
            self.use_leo or getattr(self, 'swarm_mode', False)
        ):
            raise ValueError(
                f"emr_hyperneat.weight_threshold {self.weight_threshold} would be ignored: "
                f"{'LEO gates the weights' if self.use_leo else 'the swarm build prunes'} at "
                f"{WEIGHT_THRESHOLD}"
            )
        if self.execution_config.resolve(self.max_depth) is ExecutionMode.DEPTH_OFFLOAD:
            self.refuse_depth_offload()
        if self.strategy == MultiGPUStrategy.EVAL_ONLY_PARALLEL:
            dropped = dropped + self._features_offload_drops()
            if dropped:
                raise ValueError(
                    "multi_gpu_strategy EVAL_ONLY_PARALLEL evaluates tanh hidden nodes with sum "
                    "aggregation, no neuromodulation and 1-MSE on weights it builds itself, so it "
                    "would ignore " + ", ".join(dropped)
                )

    def _refuse_pmap_pipeline_gaps(self, strategy_name: str) -> None:
        """Refuse what the pmap pipeline cannot compute, at the moment it is entered.

        The pmap pipeline (`_full_pipeline_with_hh_core` for FULL_PIPELINE_PARALLEL) takes
        hidden-to-hidden connections at iteration level 1 only: the direct connections, which the
        bounded discovery finds outside the pmap (`_pmap_pipeline_hh`). There is no multi-hop
        expansion, so a level-2 preset would run there as level 1 while the record names level 2
        (the level-2 fitness vector of the pmap route equals the standard path's level-1 vector,
        XOR depth 3 population 8 and 30). It also evaluates tanh hidden nodes with sum aggregation
        and no neuromodulation, like EVAL_ONLY_PARALLEL. This runs at dispatch, not at
        configuration time, because the same configuration falls back to the standard path on one
        device or on a small dataset, where level 2 is computed.

        Raises:
            ValueError: `allow_hidden_to_hidden` with `iteration_level` above 1, or a dropped
                feature.
        """
        recurrent = (self.extended_config is not None
                     and self.extended_config.allow_hidden_to_hidden)
        level = self.extended_config.iteration_level if self.extended_config else 0
        if recurrent and level > 1:
            raise ValueError(
                f"multi_gpu_strategy {strategy_name} takes hidden-to-hidden connections at "
                "iteration level 1 only (the direct connections, no multi-hop expansion) and would "
                f"run iteration_level {level} as level 1. Set recurrence.iteration_level: 1, or "
                "use SINGLE_GPU, EVAL_ONLY_PARALLEL or a split route (POPULATION_PMAP, "
                "POSITION_SHARDING, HYBRID), which run every level."
            )
        dropped = self._features_a_tanh_sum_path_drops() + self._features_offload_drops()
        if dropped:
            raise ValueError(
                f"multi_gpu_strategy {strategy_name} evaluates tanh hidden nodes with sum "
                "aggregation, no neuromodulation and 1-MSE on weights it builds itself, so it "
                "would ignore " + ", ".join(dropped)
            )

    def _pmap_iteration_level(self, allow_hh: bool) -> int:
        """The iteration level a pmap pipeline computes, passed to it as a static argument.

        Level 0 discovers the direct connections, as level 1 does on the standard path and
        under depth offload, so with hidden-to-hidden connections it runs as level 1: the same
        static arguments, hence the same compiled pipeline and results, and the recorded
        `effective_iteration_level` 1 holds. Passed as 0, it would read as feed-forward at the
        pipelines' ``iteration_level > 0`` gate: no hidden-to-hidden connection and the dense
        evaluation, while the record names level 1. Levels above 1 never get here
        (`_refuse_pmap_pipeline_gaps` runs first).

        Args:
            allow_hh: Whether the configuration allows hidden-to-hidden connections.

        Returns:
            The configured level, or 1 for level 0 with hidden-to-hidden connections.
        """
        level = self.extended_config.iteration_level if self.extended_config else 0
        return 1 if allow_hh and level == 0 else level

    def _check_additive_settling(self) -> None:
        """Refuse what HMR FULL's neuromodulated rule (recurrence.settling_rule 'hmr' with
        neuromodulation) cannot apply.

        Raises:
            ValueError: modulatory neurons (FULL's modulatory recurrence read one population-wide
                modulatory mask, and no runner used it), per-node hidden-to-hidden aggregation (FULL
                summed), or an every-iteration activation other than tanh (FULL's single-task update
                applies tanh).
        """
        mode = self.neuromod_config.mode
        if mode == 'modulatory_neurons':
            raise ValueError(
                "recurrence.settling_rule 'hmr' with modulatory neurons is refused: HMR FULL's "
                "modulatory recurrence read one population-wide modulatory mask, "
                "where EMR's masks are "
                "each genome's own, and no runner used it"
            )
        if self.hh_aggregation_mode == 'dynamic':
            raise ValueError(
                "recurrence.settling_rule 'hmr' with neuromodulation sums the hidden-to-hidden "
                "input, as HMR FULL did; per-node hidden-to-hidden aggregation would be ignored"
            )
        plain_tanh = self.dynamic_functions_mode == 'disabled' or (
            self.dynamic_functions_mode == 'global' and self.df_hidden_activation == 'tanh')
        if self.hh_activation_mode == 'every_iteration' and not plain_tanh:
            raise ValueError(
                "recurrence.settling_rule 'hmr' with neuromodulation applies tanh at every step "
                "after the first, as HMR FULL did; set dynamic_functions.hh_activation_mode "
                "'initial_only'"
            )

    def _features_a_tanh_sum_path_drops(self) -> List[str]:
        """The configured features an evaluation with tanh nodes, sum and no modulation drops."""
        dropped = []
        if self.dynamic_functions_mode != 'disabled':
            dropped.append(f"dynamic_functions.mode={self.dynamic_functions_mode!r}")
        if self.agg_mode != 'disabled':
            dropped.append(f"aggregation.mode={self.agg_mode!r}")
        if self.neuromod_config is not None and self.neuromod_config.mode != 'disabled':
            dropped.append("neuromodulation")
        if self._settling_rule == 'hmr':
            dropped.append("recurrence.settling_rule='hmr'")
        return dropped

    def _check_hmr_reads(self, recurrent: bool) -> None:
        """Refuse HMR's reads (cppn_read 'input_mean', weight_source 'raw') where nothing would
        apply them or where the raw input-to-node queries are not kept: they reproduce HMR's
        feed-forward dynamic-functions and aggregation variants on the standard execution path.

        Raises:
            ValueError: A read is set with recurrence, outside the standard execution mode, without
                the mode that reads it, or (input_mean) without a dedicated CPPN channel.
        """
        blocks = (
            (
                'dynamic_functions',
                self.df_cppn_read,
                self.df_weight_source,
                self.dynamic_functions_mode,
                self.dynamic_functions_mode == 'weight_interpretation'
                or (
                    self.dynamic_functions_mode == 'modular'
                    and (
                        self.df_modular_config['activation_method'] != 'none'
                        or self._df_sparsity_method == 'threshold'
                        or self._df_scaling_method != 'none'
                    )
                ),
            ),
            (
                'aggregation',
                self.agg_cppn_read,
                self.agg_weight_source,
                self.agg_mode,
                self.agg_mode in ('weight_interpretation', 'modular')
                or self.hh_aggregation_mode == 'dynamic',
            ),
        )
        for section, read, source, mode, reads_weights in blocks:
            chosen = [f"cppn_read {read!r}"] if read != 'self' else []
            chosen += [f"weight_source {source!r}"] if source != 'w1' else []
            if not chosen:
                continue
            where = f"{section} " + " and ".join(chosen)
            if recurrent:
                raise ValueError(
                    f"{where} reproduces HMR's feed-forward variants; this configuration has "
                    f"hidden-to-hidden recurrence"
                )
            if self.execution_config.resolve(self.max_depth) is not ExecutionMode.STANDARD:
                raise ValueError(
                    f"{where} needs the standard execution mode (the raw input-to-node queries are "
                    "not kept on the others)"
                )
            if read != 'self' and (mode != 'cppn_output' or self.num_cppn_outputs < 2):
                raise ValueError(
                    f"{section}.cppn_read 'input_mean' needs mode 'cppn_output' and a dedicated "
                    f"CPPN channel (HMR had no one-output read); mode {mode!r}, "
                    f"{self.num_cppn_outputs} CPPN output(s)"
                )
            if source != 'w1' and not reads_weights:
                raise ValueError(
                    f"{section}.weight_source 'raw' would be ignored: mode {mode!r} reads no "
                    "weight interpretation"
                )

    def _features_offload_drops(self) -> List[str]:
        """Configured features the evaluations that build their own weights and score 1-MSE drop.

        Depth offload, POPULATION_PARALLEL_PROCESS's workers, EVAL_ONLY_PARALLEL and the
        FULL_PIPELINE_PARALLEL pipelines score 1-MSE without the fitness rescaling and build their
        weight matrices without the locality penalty, so these settings are refused there rather
        than dropped without a word.
        """
        dropped = []
        if getattr(self, 'use_leo', False):
            dropped.append('leo')
        if getattr(self, 'fitness_mode', 'mse') != 'mse':
            dropped.append(f"fitness_mode={self.fitness_mode!r}")
        if getattr(self, 'locality_radius', None) is not None:
            dropped.append(f"locality_radius={self.locality_radius!r}")
        if getattr(self, '_fitness_rescale_variance', None) is not None:
            dropped.append(f"fitness_rescale_variance={self._fitness_rescale_variance!r}")
        if getattr(self, 'weight_threshold', WEIGHT_THRESHOLD) != WEIGHT_THRESHOLD:
            dropped.append(f"weight_threshold={self.weight_threshold!r}")
        return dropped

    def refuse_depth_offload(self) -> None:
        """Raise if the depth-offload path would ignore part of this configuration.

        Called by ``create_config`` for a configured or depth-selected ``depth_offload``, and by
        the offload generation itself, because on a GPU ``auto`` can also select it at run time
        from the population size and device memory.

        Raises:
            ValueError: Dynamic functions, aggregation, neuromodulation, a fitness mode other than
                mse, a locality radius or a fitness rescaling are configured.
        """
        dropped = self._features_a_tanh_sum_path_drops() + self._features_offload_drops()
        if dropped:
            raise ValueError(
                "execution mode 'depth_offload' evaluates tanh hidden nodes with sum "
                "aggregation, no neuromodulation and 1-MSE, without LEO, a locality penalty or a "
                "fitness rescaling, so it would ignore "
                + ", ".join(dropped)
                + ". Use execution.mode 'standard' or 'streaming' for these features."
            )

    def _parse_extended_config(self, recurrence_section: Dict[str, Any]) -> None:
        """Parse recurrence section into extended config.

        Args:
            recurrence_section: The 'recurrence' subsection of emr_hyperneat config
        """
        _refuse_unknown_keys('recurrence', recurrence_section, self.RECURRENCE_KEYS)
        # The network model: the grid, or HMR's node list, opt-in. create_config checks a node-list
        # configuration once everything is parsed (_configure_node_list).
        self.network_model = recurrence_section.get('network_model', 'grid')
        if self.network_model not in NETWORK_MODELS:
            raise ValueError(f"recurrence.network_model must be one of {NETWORK_MODELS}, got "
                             f"{self.network_model!r}")
        self._settling_rule = recurrence_section.get('settling_rule', 'emr')
        if self._settling_rule not in SETTLING_RULES:
            raise ValueError(f"recurrence.settling_rule must be one of {SETTLING_RULES}, got "
                             f"{self._settling_rule!r}")
        # Check for preset first
        preset_name = recurrence_section.get('preset', None)
        if preset_name:
            base_config = get_recurrence_preset(preset_name)
            # A preset decides whether recurrence is on; a block key that says otherwise raises,
            # since ignoring it would make the run disagree with its configuration
            for key in ('enabled', 'allow_hidden_to_hidden'):
                if key in recurrence_section and (bool(recurrence_section[key])
                                                  != bool(getattr(base_config, key))):
                    raise ValueError(
                        f"recurrence.{key}={recurrence_section[key]!r} contradicts preset "
                        f"{preset_name!r} ({key}={getattr(base_config, key)!r}): the preset "
                        f"decides it. Remove the key, or choose a preset that agrees with it.")
            if self.verbose:
                print(f"[EMR-HyperNEAT] Using preset '{preset_name}'")

            # Apply explicit overrides from recurrence_section
            # The preset provides defaults, but explicit values take precedence
            override_fields = {}
            configurable_fields = [
                'iteration_level',
                'multi_hop_algorithm',
                'hop_decay_factor',
                'hh_cache_enabled',
                'hh_refresh_interval',
                'hh_mask_change_threshold',
                'hh_cache_trigger',
                'allow_backward',
                'allow_lateral',
                'allow_self_loops',
                'activate_time',
                'max_connections',
                'max_sparse_conns',
                'use_dense_discovery',
                'hh_discovery_mode',
                'multi_hop_second_factor',
                'eval_chunk_size',  # For OOM prevention on large configs
                'sparse_cap_policy',
                'max_sparse_conns_ceiling',
            ]
            for field in configurable_fields:
                if field in recurrence_section:
                    override_fields[field] = recurrence_section[field]
            if 'hh_discovery_mode' in override_fields:
                override_fields['hh_discovery_mode'] = _validated_hh_discovery_mode(
                    override_fields['hh_discovery_mode'])
            elif 'multi_hop_second_factor' in override_fields:
                # The preset resolved its discovery mode for the masked factor at construction; a
                # factor override re-resolves it (the raw factor exists only on the union path).
                override_fields['hh_discovery_mode'] = None

            # Create new config with overrides
            if override_fields:
                from dataclasses import replace
                self.extended_config = replace(base_config, **override_fields)
                if self.verbose:
                    print(f"[EMR-HyperNEAT] Applied overrides: {override_fields}")
            else:
                self.extended_config = base_config
            if not base_config.allow_hidden_to_hidden:
                allowed = [key for key in ('allow_backward', 'allow_lateral', 'allow_self_loops')
                           if recurrence_section.get(key)]
                if allowed:
                    raise ValueError(
                        f"recurrence.{allowed[0]} is set but preset {preset_name!r} has no "
                        f"hidden-to-hidden connections, so the run would be feed-forward; choose a "
                        f"recurrent preset")
        elif recurrence_section.get('enabled', False) or recurrence_section.get('allow_hidden_to_hidden', False):
            check_recurrence_switch(recurrence_section.get('enabled', True),
                                    recurrence_section.get('allow_hidden_to_hidden', True))
            if not recurrence_section.get('allow_hidden_to_hidden', True):
                # enabled true with allow_hidden_to_hidden false builds a feed-forward network;
                # 0.1.0 kept a connection class there and ignored it
                allowed = [key for key in ('allow_backward', 'allow_lateral', 'allow_self_loops')
                           if recurrence_section.get(key)]
                if allowed:
                    raise ValueError(
                        f"recurrence.{allowed[0]} is set but allow_hidden_to_hidden is false, so "
                        "the run would be feed-forward; set allow_hidden_to_hidden: true, or drop "
                        "the flag")
            # Build config from individual settings
            self.extended_config = EMRConfig(
                enabled=recurrence_section.get('enabled', True),
                allow_hidden_to_hidden=recurrence_section.get('allow_hidden_to_hidden', True),
                allow_backward=recurrence_section.get('allow_backward', False),
                allow_lateral=recurrence_section.get('allow_lateral', False),
                allow_self_loops=recurrence_section.get('allow_self_loops', False),
                iteration_level=recurrence_section.get('iteration_level', 1),
                activate_time=recurrence_section.get('activate_time', None),
                max_connections=recurrence_section.get('max_connections', 10000),
                max_sparse_conns=recurrence_section.get('max_sparse_conns', 10000),
                sparse_cap_policy=recurrence_section.get('sparse_cap_policy', 'refuse'),
                max_sparse_conns_ceiling=recurrence_section.get('max_sparse_conns_ceiling', None),
                multi_hop_algorithm=recurrence_section.get('multi_hop_algorithm', 'matrix_power'),
                hop_decay_factor=recurrence_section.get('hop_decay_factor', 0.8),
                multi_hop_second_factor=recurrence_section.get('multi_hop_second_factor', 'masked'),
                hh_cache_enabled=recurrence_section.get('hh_cache_enabled', False),
                hh_refresh_interval=recurrence_section.get('hh_refresh_interval', 5),
                hh_mask_change_threshold=recurrence_section.get('hh_mask_change_threshold', 0.1),
                hh_cache_trigger=recurrence_section.get('hh_cache_trigger', 'time_or_mask'),
                use_dense_discovery=recurrence_section.get('use_dense_discovery', False),
                hh_discovery_mode=_validated_hh_discovery_mode(
                    recurrence_section.get('hh_discovery_mode')),
                eval_chunk_size=recurrence_section.get('eval_chunk_size', None),
            )
        elif any(recurrence_section.get(key) for key in ('allow_backward', 'allow_lateral', 'allow_self_loops')):
            allowed = [key for key in ('allow_backward', 'allow_lateral', 'allow_self_loops')
                       if recurrence_section.get(key)]
            raise ValueError(
                f"recurrence.{allowed[0]} is set but hidden-to-hidden connections are off (no "
                "preset, enabled or allow_hidden_to_hidden), so the run would be feed-forward; set "
                "a preset or allow_hidden_to_hidden: true")
        elif recurrence_section.get('eval_chunk_size') is not None:
            # Feedforward mode but with eval_chunk_size specified for OOM prevention
            self.extended_config = EMRConfig(
                enabled=False,
                allow_hidden_to_hidden=False,
                eval_chunk_size=recurrence_section.get('eval_chunk_size'),
            )
        else:
            # Feedforward mode without chunking
            self.extended_config = None

        # The cache's trigger, refused where nothing reads it
        cached = (self.extended_config is not None and self.extended_config.allow_hidden_to_hidden
                  and self.extended_config.hh_cache_enabled)
        if 'hh_cache_trigger' in recurrence_section and not cached:
            raise ValueError(
                "recurrence.hh_cache_trigger would be ignored: the h->h cache is off (it needs "
                "hidden-to-hidden connections and hh_cache_enabled)"
            )
        if (cached and self.extended_config.hh_cache_trigger == 'time'
                and 'hh_mask_change_threshold' in recurrence_section):
            raise ValueError(
                "recurrence.hh_mask_change_threshold would be ignored: hh_cache_trigger 'time' "
                "refreshes the cache every hh_refresh_interval generations only"
            )
        if (
            self.extended_config is not None
            and self.extended_config.hh_cache_trigger not in HH_CACHE_TRIGGERS
        ):
            raise ValueError(f"recurrence.hh_cache_trigger must be one of {HH_CACHE_TRIGGERS}, got "
                             f"{self.extended_config.hh_cache_trigger!r}")
        # recurrence.max_connections is parsed and checked and never read: the cap on each genome's
        # hidden-to-hidden connections is max_sparse_conns. HMR's unified model capped with
        # max_connections, its node-list model ignored it. It warns.
        if 'max_connections' in recurrence_section:
            import warnings
            warnings.warn(f"recurrence.max_connections={recurrence_section['max_connections']!r} "
                          f"has no effect: the cap on each genome's hidden-to-hidden connections "
                          f"is recurrence.max_sparse_conns (a run that reproduces an HMR "
                          f"unified-model run sets max_sparse_conns to that run's "
                          f"max_connections)", UserWarning, stacklevel=3)
        # Forward mode, iteration level and settling steps, by the rule set_recurrence_config also
        # applies. Either block sets the steps (recurrence.activate_time, substrate.activate_time),
        # and two that set them must agree; unset, a preset's own count or 2 (2**max_depth + 1)
        self._forward_mode, self.iteration_level, steps = resolve_recurrence(
            self.extended_config, self.max_depth,
            {'recurrence.activate_time': recurrence_section.get('activate_time'),
             'substrate.activate_time': self._substrate_activate_time})
        if steps is not None:
            self.activate_time = steps

        # Log configuration
        if self.extended_config is not None and self.verbose:
            mode = self._forward_mode.value
            print(f"[EMR-HyperNEAT] Mode: {mode}")
            if self.extended_config.allow_hidden_to_hidden:
                print(f"  - iteration_level: {self.extended_config.iteration_level}")
                print(f"  - multi_hop_algorithm: {self.extended_config.multi_hop_algorithm}")
                print(f"  - hop_decay_factor: {self.extended_config.hop_decay_factor}")
                print(f"  - allow_backward: {self.extended_config.allow_backward}")
                print(f"  - allow_lateral: {self.extended_config.allow_lateral}")
                print(f"  - allow_self_loops: {self.extended_config.allow_self_loops}")
                print(f"  - activate_time: {self.activate_time}")

    def _parse_dynamic_functions_config(self, hmr_config: Dict[str, Any]) -> None:
        """Parse dynamic functions and aggregation configuration.

        Args:
            hmr_config: The 'emr_hyperneat' section of config
        """
        dynamic_funcs = hmr_config.get('dynamic_functions', {})
        _refuse_unknown_keys('dynamic_functions', dynamic_funcs, self.DYNAMIC_FUNCTIONS_KEYS)
        _refuse_unknown_keys(
            'aggregation', hmr_config.get('aggregation', {}), self.AGGREGATION_KEYS
        )
        palette_evolution = dynamic_funcs.get('palette_evolution') or {}
        if not isinstance(palette_evolution, dict):
            raise ValueError(
                f"dynamic_functions.palette_evolution must be a mapping, got {palette_evolution!r}"
            )

        # Mode selection:
        # - 'disabled': Original hardcoded tanh/sigmoid (baseline)
        # - 'global': All hidden nodes use same configurable activation
        # - 'cppn_output': CPPN outputs activation index per node
        # - 'weight_interpretation': Derive activation from weight patterns
        # - 'random_fixed': Random activation per node, fixed at initialization
        # - 'random_generation': Random re-assigned each generation
        # - 'modular': Orthogonal configuration
        self.dynamic_functions_mode = dynamic_funcs.get('mode', 'disabled')
        if self.dynamic_functions_mode not in self.DYNAMIC_FUNCTION_MODES:
            raise ValueError(
                f"dynamic_functions.mode must be one of "
                f"{sorted(self.DYNAMIC_FUNCTION_MODES)}, got "
                f"{self.dynamic_functions_mode!r}. Unknown values used to fall through "
                "the activation dispatch and run the fixed tanh/sigmoid baseline "
                "instead, which reports as a successful run of the mode you asked for."
            )
        # Mode 'modular' (HMR's three orthogonal layers: activation selection, sparsity, weight
        # scaling): 0.1.0 parsed it, warned and evaluated fixed tanh.

        # For random modes: store seed for reproducibility, and the scope of the draw
        # (random_function_indices)
        self._random_mode_seed = dynamic_funcs.get('random_seed', None)
        self._random_scope = dynamic_funcs.get('random_scope', 'genome')
        if self._random_scope not in RANDOM_SCOPES:
            raise ValueError(f"dynamic_functions.random_scope must be one of {RANDOM_SCOPES}, got "
                             f"{self._random_scope!r}")
        # How a channel or a weight-interpretation signal becomes an index (INDEX_DECODES)
        self.df_index_decode = dynamic_funcs.get('index_decode', 'tanh')
        if self.df_index_decode not in INDEX_DECODES:
            raise ValueError(f"dynamic_functions.index_decode must be one of {INDEX_DECODES}, got "
                             f"{self.df_index_decode!r}")
        self.df_cppn_read, self.df_weight_source = _parse_hmr_reads(
            'dynamic_functions', dynamic_funcs
        )

        # For 'global' mode: which activation to use. Absent here, both come from the substrate
        # section (substrate.hidden_activation and substrate.output_activation), as in HMR's
        # dynamic-functions variant. An unknown name raises rather than running tanh (hidden) or
        # sigmoid (output) for it.
        self.df_hidden_activation = dynamic_funcs.get(
            'hidden_activation', getattr(self, 'hidden_activation', 'tanh'))
        self.df_output_activation = dynamic_funcs.get(
            'output_activation', getattr(self, 'output_activation', 'sigmoid'))
        for key, name in (('hidden_activation', self.df_hidden_activation),
                          ('output_activation', self.df_output_activation)):
            if name not in ACTIVATION_FUNCTIONS:
                raise ValueError(
                    f"{key} {name!r} (dynamic_functions.{key}, or substrate.{key} when that is "
                    f"absent) is not an activation function; the names are "
                    f"{sorted(ACTIVATION_FUNCTIONS)}")

        # For 'cppn_output' mode: number of activation functions available
        self.df_num_activations = dynamic_funcs.get('num_activations', 4)

        # CPPN output configuration for cppn_output mode
        # num_cppn_outputs: How many outputs the CPPN has (default from auto-compute)
        # When num_cppn_outputs=1, we derive activation from the same weight output
        # When num_cppn_outputs>1, each output can be dedicated to different purposes
        self.num_cppn_outputs = dynamic_funcs.get(
            'num_cppn_outputs', getattr(self, '_auto_num_cppn_outputs', 1)
        )

        # cppn_output_indices: Which CPPN output index to use for each purpose.
        # Channel 0 is the weight; the generation loop always reads it there.
        #
        # Aggregation defaults to the first channel nothing else reads. Before 0.2.0 a
        # two-output CPPN with aggregation (but not activation) in cppn_output mode
        # defaulted to channel 0, so aggregation was decoded from the weight and the
        # evolved second output was never read. Runs made that way are reproduced by
        # setting cppn_output_indices: {aggregation: 0}. With both modes on and an
        # explicit num_cppn_outputs of 2, channel 1 is the activation's and aggregation
        # still shares channel 0, as before.
        _refuse_unknown_keys(
            'dynamic_functions.cppn_output_indices',
            dynamic_funcs.get('cppn_output_indices'),
            self.CPPN_OUTPUT_INDICES_KEYS,
        )
        cppn_indices = dynamic_funcs.get('cppn_output_indices', {})
        if cppn_indices.get('weight', 0) != 0:
            raise ValueError(
                "dynamic_functions.cppn_output_indices.weight must be 0: the weight is read "
                "from CPPN output 0 on every path, and another value was accepted and ignored "
                "in 0.1.0"
            )
        act_from_cppn = self.dynamic_functions_mode == 'cppn_output'
        agg_from_cppn = hmr_config.get('aggregation', {}).get('mode') == 'cppn_output'
        if getattr(self, 'use_leo', False):
            # LEO holds channel LEO_CHANNEL, so the per-node outputs take the channels after
            # it: activation, then aggregation.
            if self.num_cppn_outputs <= LEO_CHANNEL:
                raise ValueError(
                    f"LEO needs a CPPN output {LEO_CHANNEL}, and num_cppn_outputs is "
                    f"{self.num_cppn_outputs}; omit num_cppn_outputs to let it be computed"
                )
            free = iter(range(LEO_CHANNEL + 1, self.num_cppn_outputs))
            default_act_channel = next(free, None) if act_from_cppn else 0
            default_agg_channel = next(free, None) if agg_from_cppn else 0
            if default_act_channel is None or default_agg_channel is None:
                raise ValueError(
                    f"num_cppn_outputs={self.num_cppn_outputs} leaves no channel for a "
                    "cppn_output mode next to the weight (0) and LEO (1); omit "
                    "num_cppn_outputs to let it be computed"
                )
        else:
            default_act_channel = 0 if self.num_cppn_outputs == 1 else 1
            if self.num_cppn_outputs <= 1:
                default_agg_channel = 0
            elif self.num_cppn_outputs == 2:
                default_agg_channel = 0 if act_from_cppn else 1
            else:
                default_agg_channel = 2
        self._cppn_output_indices = {
            'weight': cppn_indices.get('weight', 0),
            'activation': cppn_indices.get('activation', default_act_channel),
            'aggregation': cppn_indices.get('aggregation', default_agg_channel),
        }
        if getattr(self, 'use_leo', False):
            for purpose, used in (('activation', act_from_cppn), ('aggregation', agg_from_cppn)):
                if used and self._cppn_output_indices[purpose] == LEO_CHANNEL:
                    raise ValueError(
                        f"cppn_output_indices.{purpose} is {LEO_CHANNEL}, LEO's channel"
                    )

        # Validate CPPN output indices
        for purpose, idx in self._cppn_output_indices.items():
            if idx >= self.num_cppn_outputs:
                raise ValueError(
                    f"CPPN output index for '{purpose}' ({idx}) >= num_cppn_outputs "
                    f"({self.num_cppn_outputs}). "
                    f"Increase num_cppn_outputs or fix cppn_output_indices."
                )

        # H->H activation mode: how the settling steps of the recurrent path (forward mode
        # HYBRID_SPARSE_HH) activate hidden nodes; the feed-forward path is unaffected.
        # - 'every_iteration' (default): the configured activation, per-node indices or the
        #   one 'global' function, at every step.
        # - 'tanh': tanh at every step, whatever is configured.
        # 'initial_only' is rejected: the forward pass takes no initial activated step, the one
        # step its name describes, so under it no configured activation would apply on the
        # recurrent path, and a name that says the opposite of what runs is not kept.
        self.hh_activation_mode = dynamic_funcs.get('hh_activation_mode', 'every_iteration')
        # Under recurrence.settling_rule 'hmr', 'initial_only' is HMR's mode: the configured
        # activation at the first step, tanh at the later ones (settle_hidden_hmr).
        hmr_initial_only = (
            self.hh_activation_mode == 'initial_only' and self._settling_rule == 'hmr'
        )
        if self.hh_activation_mode == 'initial_only' and not hmr_initial_only:
            raise ValueError(
                "dynamic_functions.hh_activation_mode='initial_only' is no longer accepted outside "
                "recurrence.settling_rule 'hmr'. "
                "In emr-hyperneat 0.1.0 it applied no configured activation on the "
                "recurrent path: every settling step was tanh and per-node or global "
                "activations were computed and never used. Use 'every_iteration' (the "
                "default: the configured activation at every step) or 'tanh' (tanh at every "
                "step, which settles as 0.1.0 did when no input aggregation is configured)."
            )
        if self.hh_activation_mode not in HH_ACTIVATION_MODES and not hmr_initial_only:
            raise ValueError(
                f"dynamic_functions.hh_activation_mode must be one of {HH_ACTIVATION_MODES}, "
                f"got {self.hh_activation_mode!r}"
            )
        if 'global_activation' in dynamic_funcs:
            raise ValueError(
                "dynamic_functions.global_activation is not a setting; the one hidden "
                "activation of mode 'global' is dynamic_functions.hidden_activation"
            )

        # Palette configuration
        palette_config = dynamic_funcs.get('palette', None)
        if palette_config is None:
            self.df_palette = None
            self.df_palette_name = None
        elif isinstance(palette_config, str):
            # Preset name
            if palette_config in PALETTE_CONFIGS:
                self.df_palette = jnp.array(PALETTE_CONFIGS[palette_config], dtype=jnp.int32)
                self.df_palette_name = palette_config
            else:
                raise ValueError(f"Unknown palette preset: {palette_config}. "
                               f"Available: {list(PALETTE_CONFIGS.keys())}")
        else:
            # Explicit list of ACTIVATION_LIST indices or names. An index outside ACTIVATION_LIST
            # raises, because evaluation dispatches by index.
            self.df_palette = parse_function_palette(
                palette_config, ACTIVATION_LIST, 'dynamic_functions.palette'
            )
            self.df_palette_name = 'custom'

        # Built-in palette evolution (HMR's dynamic-functions variant; 0.1.0 ignored the key). It
        # evolves the palette of the per-node CPPN decode, which it owns.
        self.df_palette_evolution = None
        if palette_evolution.get('enabled', False):
            if self.dynamic_functions_mode != 'cppn_output':
                raise ValueError(
                    f"dynamic_functions.palette_evolution evolves the palette of the per-node CPPN "
                    f"decode: it needs mode 'cppn_output', got {self.dynamic_functions_mode!r}"
                )
            if palette_config is not None:
                raise ValueError(
                    "dynamic_functions.palette_evolution starts from its initial_mask: set that, "
                    "not dynamic_functions.palette"
                )
            self.df_palette_evolution = parse_palette_evolution(palette_evolution)
            self.df_palette = jnp.asarray(np.flatnonzero(self.df_palette_evolution['initial_mask']),
                                          dtype=jnp.int32)
            self.df_palette_name = 'evolving'
        elif palette_evolution:
            _refuse_unknown_keys(
                'dynamic_functions.palette_evolution', palette_evolution, PALETTE_EVOLUTION_KEYS
            )
            # A disabled block's settings are read by nothing, so they are refused
            if set(palette_evolution) - {'enabled'}:
                raise ValueError(
                    "dynamic_functions.palette_evolution settings are given but "
                    "palette_evolution.enabled "
                    "is not true, so they would be ignored"
                )

        # HMR's live palette strategy: a runner's strategy object sets the palette after each
        # generation (set_palette_strategy). HMR's perturbation wrote into TensorNEAT's node-index
        # column, a defect not reproduced; here it goes to the bias of the
        # activation output node.
        self._live_strategy = parse_live_strategy(dynamic_funcs.get('live_strategy') or {})
        self._palette_strategy = None
        if self._live_strategy is not None:
            self._check_live_strategy()

        # For 'weight_interpretation' mode: the interpretation method; an unknown name raises.
        self.df_interpretation = dynamic_funcs.get('interpretation', 'magnitude_bio')
        if self.df_interpretation not in ACTIVATION_INTERPRETATIONS:
            raise ValueError(
                f"dynamic_functions.interpretation must be one of {ACTIVATION_INTERPRETATIONS}, "
                f"got {self.df_interpretation!r}; 0.1.0 ran magnitude_bio for another name")

        # Sparsity configuration for biologically-inspired mechanisms
        sparsity_config = dynamic_funcs.get('sparsity', {})
        _refuse_unknown_keys('dynamic_functions.sparsity', sparsity_config, self.SPARSITY_KEYS)
        self.df_sparsity_config = {
            'level_0': sparsity_config.get('level_0', 0.05),
            'level_1': sparsity_config.get('level_1', 0.20),
            'level_2_plus': sparsity_config.get('level_2_plus', 0.40),
            'wta_k_percent': sparsity_config.get('wta_k_percent', 0.10),
        }

        # Critical Periods configuration
        critical_periods = dynamic_funcs.get('critical_periods', {})
        _refuse_unknown_keys('dynamic_functions.critical_periods', critical_periods,
                             self.CRITICAL_PERIODS_KEYS)
        self.df_critical_periods_enabled = critical_periods.get('enabled', False)
        if self.df_critical_periods_enabled:
            raise ValueError(
                "dynamic_functions.critical_periods is not implemented in EMR (nothing changes a "
                "mutation rate); 0.1.0 accepted it and ran without it. The critical periods of the "
                "palette paper are part of its strategies "
                "(papers/emr-dynamic-functions-bio-inspired/scripts/palette_strategies/)")
        self.df_critical_periods_config = {
            'phase1_end': critical_periods.get('phase1_end', 0.2),
            'phase2_end': critical_periods.get('phase2_end', 0.5),
            'min_plasticity': critical_periods.get('min_plasticity', 0.3),
        }

        # Modular mode configuration
        modular_config = dynamic_funcs.get('modular', {})
        _refuse_unknown_keys('dynamic_functions.modular', modular_config, self.MODULAR_KEYS)
        self.df_modular_config = {
            'activation_method': modular_config.get('activation_method', 'magnitude_bio'),
            'sparsity_method': modular_config.get('sparsity_method', 'none'),
            'scaling_method': modular_config.get('scaling_method', 'none'),
        }
        for key, names in (('activation_method', MODULAR_ACTIVATION_METHODS),
                           ('sparsity_method', MODULAR_SPARSITY_METHODS),
                           ('scaling_method', MODULAR_SCALING_METHODS)):
            if self.df_modular_config[key] not in names:
                raise ValueError(
                    f"dynamic_functions.modular.{key} must be one of {names}, got "
                    f"{self.df_modular_config[key]!r} (HMR fell back to magnitude_bio or to "
                    "nothing for an unknown name)"
                )
        modular_sparsity = modular_config.get('sparsity', {})
        _refuse_unknown_keys(
            'dynamic_functions.modular.sparsity', modular_sparsity, self.SPARSITY_KEYS
        )
        self.df_modular_sparsity = {
            'level_0': modular_sparsity.get('level_0', 0.05),
            'level_1': modular_sparsity.get('level_1', 0.20),
            'level_2_plus': modular_sparsity.get('level_2_plus', 0.40),
            'wta_k_percent': modular_sparsity.get('wta_k_percent', 0.10),
        }
        for where, values in (
            ('sparsity', self.df_sparsity_config),
            ('modular.sparsity', self.df_modular_sparsity),
        ):
            for key, value in values.items():
                if not 0.0 < float(value) <= 1.0:
                    raise ValueError(
                        f"dynamic_functions.{where}.{key} is the fraction of values kept, in (0, "
                        f"1]; got {value!r}"
                    )

        # The values are checked first (above); then whether the mode reads the blocks at all.
        # dynamic_functions.sparsity is read only as the level fractions of weight_interpretation's
        # hierarchical_sparsity, and refused anywhere else (0.1.0 parsed the block and read nothing
        # in it)
        if sparsity_config and not (self.dynamic_functions_mode == 'weight_interpretation'
                                    and self.df_interpretation == 'hierarchical_sparsity'):
            raise ValueError(
                "dynamic_functions.sparsity sets the level fractions of weight_interpretation with "
                "interpretation 'hierarchical_sparsity', the only evaluation that reads it; this "
                f"configuration has mode {self.dynamic_functions_mode!r} and interpretation "
                f"{self.df_interpretation!r}, where it would be ignored (mode 'modular' reads "
                "dynamic_functions.modular.sparsity)"
            )
        # The activation, sparsity and scaling settings are read by mode 'modular' only
        # (aggregation_method by aggregation.mode 'modular', whatever this mode), so they are
        # refused elsewhere (0.1.0 parsed the block in every mode and read nothing in it)
        modular_only = sorted(
            {'activation_method', 'sparsity_method', 'scaling_method', 'sparsity'}
            & set(modular_config)
        )
        if modular_only and self.dynamic_functions_mode != 'modular':
            raise ValueError(
                f"dynamic_functions.modular settings {modular_only} configure "
                "dynamic_functions.mode 'modular' only; this configuration has mode "
                f"{self.dynamic_functions_mode!r}, where they would be ignored (aggregation.mode "
                "'modular' reads dynamic_functions.modular.aggregation_method)"
            )

        # What the generation applies after the activation index: a sparsity method (zero the hidden
        # values below a quantile of each sample's active values) and a scaling method (scale each
        # node's incoming weights). weight_interpretation names them through its interpretation,
        # modular through its layers; each takes its own sparsity block, as in HMR.
        self._df_sparsity_method, self._df_scaling_method = 'none', 'none'
        self._df_sparsity_params = self.df_sparsity_config
        if self.dynamic_functions_mode == 'weight_interpretation':
            self._df_sparsity_method = {
                'hierarchical_sparsity': 'hierarchical',
                'sparsity_threshold': 'threshold',
            }.get(self.df_interpretation, 'none')
            if self.df_interpretation == 'stp_inspired':
                self._df_scaling_method = 'stp_inspired'
        elif self.dynamic_functions_mode == 'modular':
            self._df_sparsity_method = self.df_modular_config['sparsity_method']
            self._df_scaling_method = self.df_modular_config['scaling_method']
            self._df_sparsity_params = self.df_modular_sparsity

        # AGGREGATION FUNCTIONS CONFIGURATION
        # Note: aggregation is a sibling of dynamic_functions in the config,
        # so we read it from hmr_config, not from dynamic_funcs
        aggregation_config = hmr_config.get('aggregation', {})
        if 'global_aggregation' in aggregation_config:
            raise ValueError(
                "aggregation.global_aggregation is not a setting; the function of aggregation "
                "mode 'global' is aggregation.global_function (the misspelled key used to be "
                "ignored, so such a configuration ran 'sum')"
            )
        self.agg_mode = aggregation_config.get('mode', 'disabled')
        if self.agg_mode not in self.AGGREGATION_MODES:
            raise ValueError(
                f"aggregation.mode must be one of {sorted(self.AGGREGATION_MODES)}, got "
                f"{self.agg_mode!r}"
            )
        # Mode 'modular' takes dynamic_functions.modular.aggregation_method (HMR's aggregation
        # variant); 0.1.0 ran sum aggregation for it.

        # For random modes: the seed and the scope of the draw (random_function_indices)
        self._random_agg_seed = aggregation_config.get('random_seed', None)
        self._random_agg_scope = aggregation_config.get('random_scope', 'genome')
        if self._random_agg_scope not in RANDOM_SCOPES:
            raise ValueError(f"aggregation.random_scope must be one of {RANDOM_SCOPES}, got "
                             f"{self._random_agg_scope!r}")
        self.agg_index_decode = aggregation_config.get('index_decode', 'tanh')
        if self.agg_index_decode not in INDEX_DECODES:
            raise ValueError(f"aggregation.index_decode must be one of {INDEX_DECODES}, got "
                             f"{self.agg_index_decode!r}")
        self.agg_cppn_read, self.agg_weight_source = _parse_hmr_reads(
            'aggregation', aggregation_config
        )

        # For 'global' mode: the aggregation (an unknown name raises rather than running 'sum')
        self.agg_global_function = aggregation_config.get('global_function', 'sum')
        if self.agg_global_function not in AGGREGATION_FUNCTIONS:
            raise ValueError(
                f"aggregation.global_function must be one of {sorted(AGGREGATION_FUNCTIONS)}, got "
                f"{self.agg_global_function!r}")

        # For 'cppn_output' mode: number of aggregation functions available
        self.agg_num_aggregations = aggregation_config.get('num_aggregations', 4)
        # Aggregation palette: HMR's palette strategies computed an aggregation palette that was
        # recorded and never applied, and 0.1.0 read no agg_palette a runner set. The per-node modes
        # decode a palette position and map it to the function's AGGREGATION_LIST index, as the
        # activation palette does.
        agg_palette = aggregation_config.get('palette', None)
        self.agg_palette = (
            None
            if agg_palette is None
            else parse_function_palette(agg_palette, AGGREGATION_LIST, 'aggregation.palette')
        )

        # For 'weight_interpretation' mode: interpretation method
        self.agg_interpretation = aggregation_config.get('interpretation', 'magnitude_bio')
        if self.agg_interpretation not in AGGREGATION_INTERPRETATIONS:
            raise ValueError(
                f"aggregation.interpretation must be one of {AGGREGATION_INTERPRETATIONS}, got "
                f"{self.agg_interpretation!r}; 0.1.0 ran magnitude_bio for another name")

        # Whether to use true per-node aggregation or approximation
        self.agg_use_true_aggregation = aggregation_config.get('use_true_aggregation', True)

        # H->H aggregation mode: controls how aggregations are applied during H->H iterations
        # - 'sum': Standard scatter-add (current behavior, efficient)
        # - 'dynamic': Per-node aggregation using segment operations (more complex)
        self.hh_aggregation_mode = aggregation_config.get('hh_aggregation_mode', 'sum')
        if self.hh_aggregation_mode not in ('sum', 'dynamic'):
            raise ValueError(f"aggregation.hh_aggregation_mode must be 'sum' or 'dynamic', got "
                             f"{self.hh_aggregation_mode!r}")

        # Extend modular config to include aggregation method
        self.df_modular_config['aggregation_method'] = modular_config.get('aggregation_method', 'none')
        if self.df_modular_config['aggregation_method'] not in MODULAR_AGGREGATION_METHODS:
            raise ValueError(
                "dynamic_functions.modular.aggregation_method must be one of "
                f"{MODULAR_AGGREGATION_METHODS}, got "
                f"{self.df_modular_config['aggregation_method']!r}"
            )

        # Resolve hidden activation function for 'global' and 'disabled' modes, and for 'modular'
        # with activation method 'none', which gives every node tanh (HMR's index 0)
        modular_tanh = (self.dynamic_functions_mode == 'modular'
                        and self.df_modular_config['activation_method'] == 'none')
        if self.dynamic_functions_mode in ('disabled', 'global') or modular_tanh:
            act_name = self.df_hidden_activation if self.dynamic_functions_mode == 'global' else 'tanh'
            self._hidden_act_fn = ACTIVATION_FUNCTIONS.get(act_name, jnp.tanh)
            self._output_act_fn = ACTIVATION_FUNCTIONS.get(self.df_output_activation, jax.nn.sigmoid)
        else:
            # For cppn_output, weight_interpretation, etc., we use grouped_activation_forward
            self._hidden_act_fn = None
            self._output_act_fn = ACTIVATION_FUNCTIONS.get(self.df_output_activation, jax.nn.sigmoid)

        # Log configuration
        if self.verbose:
            if self.dynamic_functions_mode != 'disabled':
                print(f"[Dynamic Functions] Mode: {self.dynamic_functions_mode}")
                if self.dynamic_functions_mode == 'global':
                    print(f"  - Hidden activation: {self.df_hidden_activation}")
                elif self.dynamic_functions_mode == 'cppn_output':
                    print(f"  - Num activations: {self.df_num_activations}")
                    print(f"  - Num CPPN outputs: {self.num_cppn_outputs}")
                    print(f"  - CPPN output indices: {self._cppn_output_indices}")
                    if self.df_palette_name:
                        print(f"  - Palette: {self.df_palette_name} ({len(self.df_palette)} functions)")
                elif self.dynamic_functions_mode == 'weight_interpretation':
                    print(f"  - Interpretation: {self.df_interpretation}")
                elif self.dynamic_functions_mode == 'modular':
                    print(f"  - Activation method: {self.df_modular_config['activation_method']}")
                    print(f"  - Sparsity method: {self.df_modular_config['sparsity_method']}")
                    print(f"  - Scaling method: {self.df_modular_config['scaling_method']}")
                print(f"  - H->H activation mode: {self.hh_activation_mode}")

            if self.agg_mode != 'disabled':
                print(f"[Aggregation Functions] Mode: {self.agg_mode}")
                if self.agg_mode == 'global':
                    print(f"  - Global function: {self.agg_global_function}")
                elif self.agg_mode == 'cppn_output':
                    print(f"  - Num aggregations: {self.agg_num_aggregations}")
                elif self.agg_mode == 'weight_interpretation':
                    print(f"  - Interpretation: {self.agg_interpretation}")
                print(f"  - True aggregation: {self.agg_use_true_aggregation}")
                print(f"  - H->H aggregation mode: {self.hh_aggregation_mode}")

    def _parse_neuromodulation_config(self, hmr_config: Dict[str, Any]) -> None:
        """Parse neuromodulation and multi-task configuration.

        Supports 4 neuromodulation levels:
        1. static_gating: one gate in [0, 1] per hidden node from its input weights, or per
           connection from a dedicated CPPN output (gate_source 'cppn')
        2. context_gating: XdG-style context modulates gates
        3. modulatory_neurons: some of a genome's own nodes set a per-case gain for the others
           (activity modulation, not Soltoggio's plasticity gating)
        4. true_neuromodulation: NT vectors + receptor densities + gain modulation

        Args:
            hmr_config: The 'emr_hyperneat' section of config
        """
        raw_neuromodulation = hmr_config.get('neuromodulation', {})
        neuromod_config = _translate_hmr_neuromodulation(raw_neuromodulation)
        _refuse_unknown_keys('neuromodulation', neuromod_config, self.NEUROMODULATION_KEYS)
        # The modulatory CPPN outputs the modulatory options read (mode modulatory_neurons; set
        # below)
        self._modulatory_cppn_reads: Tuple[str, ...] = ()

        # Mode selection (default: disabled = original behavior). An unknown mode raises here, at
        # configuration, rather than at the first generation's dispatch.
        mode = neuromod_config.get('mode', 'disabled')
        if mode not in self.NEUROMODULATION_MODES:
            raise ValueError(
                f"neuromodulation.mode must be one of {self.NEUROMODULATION_MODES}, got {mode!r}; "
                "HMR's neuromodulation variant used preset names as modes (e.g. "
                "'true_neuromodulation_4nt_option_a_tanh', 'xdg_style', 'full', 'custom'); EMR's "
                "presets are neuromodulation.preset")
        if neuromod_config.get('branch_gating', False):
            raise ValueError("neuromodulation.branch_gating is not implemented (no evaluation applies "
                             "it; the multi-task routing is multitask.branch_gating_mode); 0.1.0 "
                             "accepted and ignored it")
        if neuromod_config.get('context_source', 'task_id') != 'task_id':
            raise ValueError(
                f"neuromodulation.context_source must be 'task_id', got "
                f"{neuromod_config['context_source']!r}: context gating takes its context from the "
                f"current task index, and no evaluation reads another source (0.1.0 "
                f"accepted 'input' and 'learned' and ignored them)")

        # Check for preset
        preset = neuromod_config.get('preset', None)
        if preset is not None and preset not in NEUROMODULATION_PRESETS:
            raise ValueError(f"neuromodulation.preset must be one of {sorted(NEUROMODULATION_PRESETS)}, "
                             f"got {preset!r}; 0.1.0 ignored an unknown preset")
        levels_named = (preset not in (None, 'disabled')) or mode != 'disabled'
        if neuromod_config.get('enabled', False) and not levels_named:
            raise ValueError("neuromodulation.enabled is true but no mode or preset is set, so "
                             "modulation would be off")
        given = raw_neuromodulation if isinstance(raw_neuromodulation, dict) else {}
        if 'enabled' in given and not given['enabled'] and levels_named:
            raise ValueError("neuromodulation.enabled is false but a mode or preset is set; 0.1.0 ran "
                             "the mode whatever enabled said. Set mode 'disabled' or drop enabled")
        if preset:
            base = get_neuromodulation_config(preset)
            # The preset's level, as the mode the evaluation dispatches on. The presets set level
            # flags and no mode, so without this a preset would leave the mode 'disabled' and the
            # run would have no modulation.
            levels = tuple(name for name in ('static_gating', 'context_gating', 'modulatory_neurons',
                                             'true_neuromodulation') if getattr(base, name))
            preset_mode = {(): 'disabled', ('static_gating',): 'static_gating',
                           ('static_gating', 'context_gating'): 'context_gating',
                           ('modulatory_neurons',): 'modulatory_neurons',
                           ('true_neuromodulation',): 'true_neuromodulation'}.get(levels)
            if preset_mode is None:
                raise ValueError(f"neuromodulation.preset {preset!r} combines the levels {levels}; "
                                 f"HMR evaluated {levels[-1]} alone while its CPPN also evolved "
                                 f"the outputs of the others, which nothing read, so EMR runs one "
                                 f"level: configure {levels[-1]}")
            if 'mode' in neuromod_config and mode != preset_mode:
                raise ValueError(f"neuromodulation.mode {mode!r} contradicts preset {preset!r} (level "
                                 f"{preset_mode!r})")
            # Override with explicit settings
            self.neuromod_config = NeuromodulationConfig(
                enabled=preset_mode != 'disabled',
                static_gating=base.static_gating,
                context_gating=base.context_gating,
                modulatory_neurons=base.modulatory_neurons,
                true_neuromodulation=base.true_neuromodulation,
                mode=preset_mode,
                # Static gating
                gate_threshold=neuromod_config.get('gate_threshold', base.gate_threshold),
                gate_scaling=neuromod_config.get('gate_scaling', base.gate_scaling),
                gate_hardness=neuromod_config.get('gate_hardness', base.gate_hardness),
                # Context gating (XdG-style)
                context_dim=neuromod_config.get('context_dim', base.context_dim),
                context_influence=neuromod_config.get('context_influence', base.context_influence),
                context_source=neuromod_config.get('context_source', base.context_source),
                # Modulatory neurons (activity modulation, not plasticity)
                mod_neuron_ratio=neuromod_config.get('mod_neuron_ratio', base.mod_neuron_ratio),
                mod_connection_type=neuromod_config.get(
                    'mod_connection_type', base.mod_connection_type
                ),
                mod_decay=neuromod_config.get('mod_decay', base.mod_decay),
                # TRUE neuromodulation
                num_nt_types=neuromod_config.get('num_nt_types', base.num_nt_types),
                modulation_strength=neuromod_config.get(
                    'modulation_strength', base.modulation_strength
                ),
                receptor_from_weight=neuromod_config.get(
                    'receptor_from_weight', base.receptor_from_weight
                ),
                receptor_derivation=neuromod_config.get(
                    'receptor_derivation', base.receptor_derivation
                ),
                modulation_mode=neuromod_config.get('modulation_mode', base.modulation_mode),
                use_output_inversion=neuromod_config.get(
                    'use_output_inversion', base.use_output_inversion
                ),
                node_rule=neuromod_config.get('node_rule', base.node_rule),
                # Branch-specific gating
                branch_gating=neuromod_config.get('branch_gating', base.branch_gating),
                # Read on the preset path too
                use_self_connection_query=neuromod_config.get('use_self_connection_query', False),
                gate_source=neuromod_config.get(
                    'gate_source', 'cppn' if neuromod_config.get('gate_from_cppn') else 'weights'
                ),
                modulatory_source=neuromod_config.get('modulatory_source', 'weights'),
                # The modulatory options the block sets; the others take the source's bundle
                **{
                    key: neuromod_config[key]
                    for key in (*MODULATORY_OPTIONS, 'mod_neuron_count')
                    if key in neuromod_config
                },
            )
            if preset_mode == 'disabled':
                self.neuromod_config = None
        elif mode != 'disabled':
            # Build config from individual settings
            # Set mode-specific flags based on the mode string
            self.neuromod_config = NeuromodulationConfig(
                enabled=True,  # Any non-disabled mode is enabled
                mode=mode,
                # Mode-specific flags
                static_gating=(mode == 'static_gating'),
                context_gating=(mode == 'context_gating'),
                modulatory_neurons=(mode == 'modulatory_neurons'),
                true_neuromodulation=(mode == 'true_neuromodulation'),
                # Static gating params
                gate_threshold=neuromod_config.get('gate_threshold', 0.5),
                gate_scaling=neuromod_config.get('gate_scaling', 'sigmoid'),
                gate_hardness=neuromod_config.get('gate_hardness', 10.0),
                # Context gating (XdG-style)
                context_dim=neuromod_config.get('context_dim', 4),
                context_influence=neuromod_config.get('context_influence', 0.5),
                context_source=neuromod_config.get('context_source', 'task_id'),
                # Modulatory neurons (activity modulation, not plasticity)
                mod_neuron_ratio=neuromod_config.get('mod_neuron_ratio', 0.1),
                mod_connection_type=neuromod_config.get('mod_connection_type', 'multiplicative'),
                mod_decay=neuromod_config.get('mod_decay', 0.9),
                # TRUE neuromodulation
                num_nt_types=neuromod_config.get('num_nt_types', 4),
                modulation_strength=neuromod_config.get('modulation_strength', 2.0),
                receptor_from_weight=neuromod_config.get('receptor_from_weight', True),
                receptor_derivation=neuromod_config.get('receptor_derivation', 'tanh'),
                modulation_mode=neuromod_config.get('modulation_mode', 'full'),
                use_output_inversion=neuromod_config.get('use_output_inversion', True),
                node_rule=neuromod_config.get('node_rule', 'paper'),
                # Branch-specific gating
                branch_gating=neuromod_config.get('branch_gating', False),
                # Opt-in self-connection-query decode (reproduces the HMR/paper
                # receptor/base-gain source). Default False = existing EMR behavior.
                use_self_connection_query=neuromod_config.get('use_self_connection_query', False),
                gate_source=neuromod_config.get(
                    'gate_source', 'cppn' if neuromod_config.get('gate_from_cppn') else 'weights'
                ),
                modulatory_source=neuromod_config.get('modulatory_source', 'weights'),
                # The modulatory options the block sets; the others take the source's bundle
                **{
                    key: neuromod_config[key]
                    for key in (*MODULATORY_OPTIONS, 'mod_neuron_count')
                    if key in neuromod_config
                },
            )
        else:
            # Disabled mode - original behavior
            self.neuromod_config = None

        # Neuromodulation's dedicated CPPN outputs: Option B's receptors, the per-connection gate,
        # or the modulatory factor and node type, the CPPN's last channels after the weight, LEO and
        # the per-node channels (HMR's order; the output count includes them)
        if self.neuromod_config:
            for key, value in (('gate_source', self.neuromod_config.gate_source),
                               ('modulatory_source', self.neuromod_config.modulatory_source)):
                if value not in NEUROMODULATION_SOURCES:
                    raise ValueError(
                        f"neuromodulation.{key} must be one of {NEUROMODULATION_SOURCES}, got "
                        f"{value!r}"
                    )
            mode = self.neuromod_config.mode
            if self.neuromod_config.gate_source == 'cppn' and mode not in (
                'static_gating',
                'context_gating',
            ):
                raise ValueError(
                    "neuromodulation.gate_source 'cppn' gates connections in modes static_gating "
                    f"and context_gating, not {mode!r}"
                )
            if self.neuromod_config.modulatory_source == 'cppn' and mode != 'modulatory_neurons':
                raise ValueError(
                    f"neuromodulation.modulatory_source 'cppn' chooses modulatory neurons in mode "
                    f"modulatory_neurons, not {mode!r}"
                )
            # The node rule is true neuromodulation's: another mode would ignore it
            rule = self.neuromod_config.node_rule
            if rule != 'paper' and mode != 'true_neuromodulation':
                raise ValueError(
                    f"neuromodulation.node_rule {rule!r} is a rule of mode true_neuromodulation; "
                    f"mode {mode!r} would ignore it"
                )
            if rule == 'anti':
                # HMR ANTI's rule gates and scales every node from its own modulation and inverts no
                # output
                if self.neuromod_config.modulation_mode != 'full':
                    raise ValueError(
                        "neuromodulation.node_rule 'anti' gates and scales every node (HMR ANTI "
                        "had no other form); modulation_mode "
                        f"{self.neuromod_config.modulation_mode!r} is not one of its forms"
                    )
                if neuromod_config.get('use_output_inversion', False):
                    raise ValueError(
                        "neuromodulation.use_output_inversion would be ignored: node_rule 'anti' "
                        "(HMR ANTI's rule) inverts no output"
                    )
                self.neuromod_config.use_output_inversion = False
            # The modulatory-neuron options: checked in their mode, refused in any other
            options_set = [
                key for key in (*MODULATORY_OPTIONS, 'mod_neuron_count') if key in neuromod_config
            ]
            if mode == 'modulatory_neurons':
                check_modulatory_options(self.neuromod_config, neuromod_config)
                self._modulatory_cppn_reads = modulatory_cppn_reads(vars(self.neuromod_config))
            elif options_set:
                raise ValueError(
                    f"neuromodulation.{options_set[0]} is an option of mode modulatory_neurons, "
                    f"not {mode!r}"
                )
            channels = neuromodulation_cppn_channels(hmr_config.get('neuromodulation', {}))
            if channels:
                base = self.num_cppn_outputs - len(channels)
                used = (
                    [0]
                    + ([LEO_CHANNEL] if getattr(self, 'use_leo', False) else [])
                    + [
                        self._cppn_output_indices[purpose]
                        for purpose, on in (
                            ('activation', self.dynamic_functions_mode == 'cppn_output'),
                            ('aggregation', getattr(self, 'agg_mode', 'disabled') == 'cppn_output'),
                        )
                        if on
                    ]
                )
                if base <= max(used):
                    raise ValueError(
                        f"neuromodulation reads {len(channels)} dedicated CPPN outputs "
                        f"({channels[0]}), the CPPN's last channels after the weight and the "
                        f"per-node channels; num_cppn_outputs {self.num_cppn_outputs} leaves no "
                        "room: omit num_cppn_outputs to let it be computed"
                    )
                for offset, name in enumerate(dict.fromkeys(channels)):
                    self._cppn_output_indices[name] = base + offset

        # Update TRUE neuromodulation storage if enabled
        if self.neuromod_config and self.neuromod_config.mode == 'true_neuromodulation':
            self._neuromod_true = {
                'receptor_densities': None,  # Set during substrate building
                'base_gains': None,  # Set during substrate building
                'num_nt_types': self.neuromod_config.num_nt_types,
                'modulation_strength': self.neuromod_config.modulation_strength,
                'option': 'A' if self.neuromod_config.receptor_from_weight else 'B',
            }

        # Multi-task configuration
        multitask_config = hmr_config.get('multitask', {})
        _refuse_unknown_keys('multitask', multitask_config, self.MULTITASK_KEYS)
        # Settings nothing in EMR applies, or that cannot rank genomes, raise unless left at their
        # default. The two constant bonuses add one value to every genome (orthogonality_bonus from the
        # tasks' NT vectors, specialization_bonus from the population's mean fitness variance): it never
        # changes a rank within a generation, only the reported fitness (and with it stagnation and a
        # threshold stop); the per-genome bonuses are orthogonality_bonus_weight and
        # specialization_bonus_weight (HMR's mechanisms). Nothing reads confusion_eval_frequency.
        for key, inert, why in (
            (
                'orthogonality_bonus',
                0.0,
                "adds one constant to every genome (computed from the tasks' NT vectors), so it "
                "never changes a rank within a generation; the per-genome bonus is "
                "multitask.orthogonality_bonus_weight",
            ),
            (
                'specialization_bonus',
                0.0,
                "adds one constant to every genome (the population's mean variance), so "
                "it never changes a rank within a generation; the "
                "per-genome bonus is "
                "multitask.specialization_bonus_weight",
            ),
            ('confusion_eval_frequency', 0, "is never read"),
        ):
            if multitask_config.get(key, inert) != inert:
                raise ValueError(f"multitask.{key}={multitask_config[key]!r} {why}")
        for key in ('orthogonality_bonus_weight', 'specialization_bonus_weight'):
            value = multitask_config.get(key, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"multitask.{key} must be a number at least 0, got {value!r}")
        if multitask_config.get('orthogonality_metric', 'cosine_mean') not in ORTHOGONALITY_METRICS:
            raise ValueError(
                f"multitask.orthogonality_metric must be one of {ORTHOGONALITY_METRICS}, got "
                f"{multitask_config['orthogonality_metric']!r}"
            )
        if (multitask_config.get('orthogonality_metric', 'cosine_mean') != 'cosine_mean'
                and not multitask_config.get('orthogonality_bonus_weight', 0.0)):
            raise ValueError(
                "multitask.orthogonality_metric applies only with "
                "multitask.orthogonality_bonus_weight "
                "above 0, so it would not change the run"
            )
        if multitask_config and not multitask_config.get('enabled', False) and any(
                key != 'enabled' for key in multitask_config):
            raise ValueError("multitask settings are given but multitask.enabled is not true, so they "
                             "would be ignored")
        if multitask_config.get('enabled', False):
            for key, value, names in (
                    ('fitness_mode', multitask_config.get('fitness_mode', 'mse'), MULTITASK_FITNESS_MODES),
                    ('fitness_aggregation', multitask_config.get('fitness_aggregation', 'mean'),
                     MULTITASK_AGGREGATIONS),
                    ('generalist_bonus_type', multitask_config.get('generalist_bonus_type', 'none'),
                     GENERALIST_BONUS_TYPES),
                    ('two_module_mode', multitask_config.get('two_module_mode', 'none'), TWO_MODULE_MODES),
                    ('modulation_mode', multitask_config.get('modulation_mode', 'full'), MODULATION_MODES)):
                if value not in names:
                    raise ValueError(f"multitask.{key} must be one of {names}, got {value!r}")
            # The hidden activations the multi-task evaluation runs: an unknown name raises rather
            # than running tanh
            per_task = multitask_config.get('per_task_activation') or {}
            for name in [multitask_config.get('hidden_activation', 'tanh'), *per_task.values()]:
                if name not in ACTIVATION_FUNCTIONS:
                    raise ValueError(f"unknown multi-task activation {name!r}; the names are "
                                     f"{sorted(ACTIVATION_FUNCTIONS)}")
            check_branch_gating_and_penalty(
                multitask_config.get('branch_gating_mode', 'none'),
                multitask_config.get('modulation_penalty', 0.0),
                multitask_config.get('two_module_mode', 'none'),
                self.neuromod_config.num_nt_types if self.neuromod_config is not None else None,
                multitask_config.get('num_tasks', 2),
            )
            self.multitask_config = MultiTaskConfig(
                enabled=True,
                num_tasks=multitask_config.get('num_tasks', 2),
                task_names=multitask_config.get('task_names', None),
                nt_per_task=multitask_config.get('nt_per_task', None),
                fitness_aggregation=multitask_config.get('fitness_aggregation', 'mean'),
                task_weights=multitask_config.get('task_weights', None),
                joint_evolution=multitask_config.get('joint_evolution', True),
                orthogonality_bonus=multitask_config.get('orthogonality_bonus', 0.0),
                specialization_bonus=multitask_config.get('specialization_bonus', 0.0),
                # Dynamic activation support for per-task activation functions
                hidden_activation=multitask_config.get('hidden_activation', 'tanh'),
                per_task_activation=multitask_config.get('per_task_activation', None),
                # Read by the multi-task evaluation, so a runner can set these keys in the
                # configuration as well as on the MultiTaskConfig object
                fitness_mode=multitask_config.get('fitness_mode', 'mse'),
                modulation_mode=multitask_config.get('modulation_mode', 'full'),
                modulation_strength_override=multitask_config.get(
                    'modulation_strength_override', None
                ),
                two_module_mode=multitask_config.get('two_module_mode', 'none'),
                generalist_bonus_type=multitask_config.get('generalist_bonus_type', 'none'),
                generalist_bonus_weight=multitask_config.get('generalist_bonus_weight', 0.0),
                generalist_threshold=multitask_config.get('generalist_threshold', 0.9),
                softmin_temperature=multitask_config.get('softmin_temperature', 0.1),
                # The per-genome bonuses
                orthogonality_bonus_weight=float(
                    multitask_config.get('orthogonality_bonus_weight', 0.0)
                ),
                orthogonality_metric=multitask_config.get('orthogonality_metric', 'cosine_mean'),
                specialization_bonus_weight=float(
                    multitask_config.get('specialization_bonus_weight', 0.0)
                ),
                # HMR's branch gating and modulation penalty
                branch_gating_mode=multitask_config.get('branch_gating_mode', 'none'),
                modulation_penalty=float(multitask_config.get('modulation_penalty', 0.0)),
            )

            # Initialize NT vectors if true_neuromodulation is enabled
            if (self.neuromod_config and
                self.neuromod_config.mode == 'true_neuromodulation'):
                num_tasks = self.multitask_config.num_tasks
                num_nt = self.neuromod_config.num_nt_types

                # Get NT vectors for each task
                if self.multitask_config.nt_per_task is not None:
                    # Use user-specified vectors
                    nt_vectors = jnp.array(self.multitask_config.nt_per_task)
                else:
                    # Use default presets
                    task_names = (self.multitask_config.task_names or
                                  [f'task_{i}' for i in range(num_tasks)])
                    nt_vectors = jnp.array([
                        get_nt_for_task(name, num_nt) for name in task_names
                    ])

                self._multitask_state = {
                    'nt_vectors': nt_vectors,
                    'task_fitnesses': None,
                    'task_names': (self.multitask_config.task_names or
                                   [f'task_{i}' for i in range(num_tasks)]),
                    'current_task_idx': 0,
                }
        else:
            self.multitask_config = None

        # Log neuromodulation configuration
        if self.verbose and self.neuromod_config:
            mode = self.neuromod_config.mode
            print(f"[Neuromodulation] Mode: {mode}")
            if mode == 'static_gating':
                print(f"  - Gate threshold: {self.neuromod_config.gate_threshold}")
                print(f"  - Gate scaling: {self.neuromod_config.gate_scaling}")
            elif mode == 'context_gating':
                print(f"  - Context dim: {self.neuromod_config.context_dim}")
                print(f"  - Context influence: {self.neuromod_config.context_influence}")
                print(f"  - Context source: {self.neuromod_config.context_source}")
            elif mode == 'modulatory_neurons':
                print(f"  - Mod neuron ratio: {self.neuromod_config.mod_neuron_ratio}")
                print(f"  - Connection type: {self.neuromod_config.mod_connection_type}")
                print(f"  - Decay: {self.neuromod_config.mod_decay}")
                print("  - Options: " + ", ".join(f"{key}={getattr(self.neuromod_config, key)}"
                                              for key in (*MODULATORY_OPTIONS, 'mod_neuron_count')))
            elif mode == 'true_neuromodulation':
                print(f"  - Num NT types: {self.neuromod_config.num_nt_types}")
                print(f"  - Modulation strength: {self.neuromod_config.modulation_strength}")
                print(f"  - Receptor from weight: {self.neuromod_config.receptor_from_weight}")
                print(f"  - Receptor derivation: {self.neuromod_config.receptor_derivation}")
                print(f"  - Modulation mode: {self.neuromod_config.modulation_mode}")
                print(f"  - Output inversion: {self.neuromod_config.use_output_inversion}")

            if self.multitask_config:
                print("[Multi-Task] Enabled")
                print(f"  - Num tasks: {self.multitask_config.num_tasks}")
                print(f"  - Fitness aggregation: {self.multitask_config.fitness_aggregation}")
                if self.multitask_config.orthogonality_bonus > 0:
                    print(f"  - Orthogonality bonus: {self.multitask_config.orthogonality_bonus}")
                if self.multitask_config.specialization_bonus > 0:
                    print(f"  - Specialization bonus: {self.multitask_config.specialization_bonus}")
                # Show activation function configuration
                if self.multitask_config.per_task_activation:
                    print(f"  - Per-task activations: {self.multitask_config.per_task_activation}")
                elif self.multitask_config.hidden_activation != 'tanh':
                    print(f"  - Hidden activation: {self.multitask_config.hidden_activation}")

    def get_recurrence_mode(self) -> str:
        """Get current recurrence mode as string ('node_list' under the node-list model)."""
        if self.network_model == 'node_list':
            return 'node_list'
        if self.extended_config is None:
            return 'feedforward'
        elif self.extended_config.enabled and (
            self.extended_config.allow_backward or
            self.extended_config.allow_lateral or
            self.extended_config.allow_self_loops
        ):
            return 'full_recurrent'
        elif self.extended_config.allow_hidden_to_hidden:
            return 'feedforward+hh'
        else:
            return 'feedforward'

    def get_extended_config(self) -> Optional[EMRConfig]:
        """Get current extended configuration."""
        return self.extended_config

    def set_recurrence_config(self, config: 'EMRConfig') -> None:
        """Enable recurrence imperatively, with the forward mode, iteration level and settling steps the
        configuration would give (`resolve_recurrence`).

        Kept because benchmark scripts drive recurrence this way. It writes
        `extended_config`, the same field `create_config` writes, so a run that
        configures recurrence through YAML and one that calls this agree. It sets
        `activate_time` and `iteration_level` as the configuration does, so a
        generation settles the configured count, and it chooses the forward mode by
        ``allow_hidden_to_hidden``, as the configuration does, not by
        ``enabled and iteration_level > 0``. Called before `create_config`, it is
        replaced by the configuration's own recurrence settings.

        Args:
            config: Recurrence settings; only ``allow_hidden_to_hidden``, ``iteration_level``
                and ``activate_time`` decide the mode, level and steps. A None ``activate_time``
                takes an explicit ``substrate.activate_time`` from the configuration, else
                ``2 (2**max_depth + 1)``; before `create_config`, when the depth is unknown,
                it derives nothing.

        Raises:
            ValueError: With hidden-to-hidden connections on, a count that is not an integer of
                at least 1, or one that differs from the configuration's
                ``substrate.activate_time``; ``enabled`` false with ``allow_hidden_to_hidden``
                true (`check_recurrence_switch`; an object without ``enabled`` counts as enabled);
                any call under network_model 'node_list'.
        """
        if self.network_model == 'node_list':
            # The node list's settings are checked once, in create_config
            raise ValueError("set_recurrence_config does not apply to recurrence.network_model "
                             "'node_list': set its recurrence in the configuration")
        check_recurrence_switch(getattr(config, 'enabled', True), config.allow_hidden_to_hidden)
        mode, level, steps = resolve_recurrence(
            config, self.max_depth,
            {"set_recurrence_config's activate_time": config.activate_time,
             'substrate.activate_time': self._substrate_activate_time})
        self.extended_config = config
        self._forward_mode = mode
        self.iteration_level = level
        if steps is not None:
            self.activate_time = steps

    def set_sparse_cap(self, max_sparse_conns: Any = None, sparse_cap_policy: Any = None,
                       max_sparse_conns_ceiling: Any = _KEEP_CEILING) -> None:
        """Change the h->h cap, its policy or its ceiling, from the next discovery on.

        The run's configuration changes in place, so every route's next discovery reads it;
        `run_until_threshold` records the cap a run ends with (``result['sparse_cap']``).

        Args:
            max_sparse_conns: The cap (None: unchanged).
            sparse_cap_policy: 'refuse', 'truncate' or 'grow' (None: unchanged).
            max_sparse_conns_ceiling: The most 'grow' may reach, or None for no ceiling (left out:
                unchanged).

        Raises:
            ValueError: Without hidden-to-hidden connections, under network_model 'node_list', with
                nothing to change, or with settings `check_sparse_cap_settings` refuses.
        """
        if self.network_model == 'node_list':
            raise ValueError(
                "set_sparse_cap does not apply to recurrence.network_model 'node_list', which has "
                "no max_sparse_conns")
        config = self.extended_config
        if config is None or not getattr(config, 'allow_hidden_to_hidden', False):
            raise ValueError(
                "set_sparse_cap needs a configuration with hidden-to-hidden connections")
        changes: Dict[str, Any] = {}
        if max_sparse_conns is not None:
            changes['max_sparse_conns'] = max_sparse_conns
        if sparse_cap_policy is not None:
            changes['sparse_cap_policy'] = sparse_cap_policy
        if max_sparse_conns_ceiling is not _KEEP_CEILING:
            changes['max_sparse_conns_ceiling'] = max_sparse_conns_ceiling
        if not changes:
            raise ValueError("set_sparse_cap: give max_sparse_conns, sparse_cap_policy or "
                             "max_sparse_conns_ceiling")
        checked = dataclasses.replace(config, **changes)  # EMRConfig.__post_init__ checks them
        with _sparse_cap_lock:
            for key in changes:
                setattr(config, key, getattr(checked, key))

    def get_forward_mode(self) -> ForwardPassMode:
        """Current forward-pass mode."""
        return self._forward_mode

    def set_forward_mode(self, mode: ForwardPassMode) -> None:
        """Set the forward-pass mode directly.

        Args:
            mode: DENSE_ONLY for feedforward, HYBRID_SPARSE_HH for recurrent.

        Raises:
            NotImplementedError: For FULL_SPARSE, which is not implemented: the generation loop
                would run it as DENSE_ONLY.
        """
        if mode is ForwardPassMode.FULL_SPARSE:
            raise NotImplementedError(
                "ForwardPassMode.FULL_SPARSE is not implemented: 0.1.0 accepted it and ran it "
                "with no h->h connections (as DENSE_ONLY). Use HYBRID_SPARSE_HH for recurrence.")
        self._forward_mode = mode
