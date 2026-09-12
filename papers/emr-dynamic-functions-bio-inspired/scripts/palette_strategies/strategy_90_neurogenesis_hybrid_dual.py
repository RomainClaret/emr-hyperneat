"""Strategy 90: Neurogenesis + Tag-Homeostatic Hybrid.

Combines tag_homeostatic_dual with adult neurogenesis:
- Base: Strategy 84 (Tag+Homeostatic) - 67% Parity-5 solve, 100% sin retention
- Extension: Strategy 63 (Adult Neurogenesis) - Birth, maturation, survival

Key innovation: Young neurons have higher tagging sensitivity but cannot be
captured until mature. This allows controlled exploration while protecting
valuable discovered functions through the tag-capture mechanism.

Bio inspiration: Adult neurogenesis in hippocampus creates new neurons that
go through maturation before integration. Combined with synaptic tagging,
only functionally important neurons survive and get captured.

Expected: Better exploration (neurogenesis) with stable retention (tag+homeostatic).
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import jax
import jax.numpy as jnp

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
    AVERAGING_AGGS,
    EXTREME_AGGS,
    CORE_EXTREME_AGGS,
)


class NeurogenesisHybridDualStrategy(PaletteEvolutionStrategy):
    """Tag-Homeostatic base with adult neurogenesis extension.

    Hybrid combining:
    - Tag+Homeostatic (84): Tag-and-capture + homeostatic balance
    - Adult Neurogenesis (63): Birth, maturation, survival dynamics

    Critical interaction: Young neurons have 2x tagging sensitivity but
    cannot be captured until mature. This balances exploration with retention.
    """

    name = "neurogenesis_hybrid_dual"
    description = "Dual: Tag+Homeostatic base with adult neurogenesis"

    def __init__(
        self,
        # === Neurogenesis parameters (from strategy 63) ===
        neurogenesis_rate: float = 0.08,
        maturation_period: int = 10,
        young_plasticity: float = 2.0,
        survival_threshold: float = 0.1,
        max_young_act: int = 3,
        max_young_agg: int = 2,
        contribution_decay: float = 0.9,
        contribution_boost: float = 0.3,
        cross_survival_boost: float = 0.15,
        # === Tagging parameters (from strategy 84/81) ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        young_tag_boost: float = 2.0,  # NEW: Young neurons tag more easily
        # === Homeostatic parameters (from strategy 84/82) ===
        target_extreme_ratio: float = 0.60,
        imbalance_threshold: float = 0.15,
        correction_strength: float = 1.8,
        discovery_bonus: float = 0.5,
        extreme_protection: float = 0.6,
        sin_protection: float = 0.5,
        # === Cross-domain parameters ===
        cross_learning_rate: float = 0.08,
        sin_extreme_affinity_boost: float = 0.3,
        # === Constraints ===
        min_stable_act: int = 2,
        min_stable_agg: int = 1,
        max_stable_act: int = 8,
        max_stable_agg: int = 4,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Neurogenesis+Tag+Homeostatic hybrid strategy."""
        # Neurogenesis
        self.neurogenesis_rate = neurogenesis_rate
        self.maturation_period = maturation_period
        self.young_plasticity = young_plasticity
        self.survival_threshold = survival_threshold
        self.max_young_act = max_young_act
        self.max_young_agg = max_young_agg
        self.contribution_decay = contribution_decay
        self.contribution_boost = contribution_boost
        self.cross_survival_boost = cross_survival_boost

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost
        self.young_tag_boost = young_tag_boost

        # Homeostatic
        self.target_extreme_ratio = target_extreme_ratio
        self.imbalance_threshold = imbalance_threshold
        self.correction_strength = correction_strength
        self.discovery_bonus = discovery_bonus
        self.extreme_protection = extreme_protection
        self.sin_protection = sin_protection

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.sin_extreme_affinity_boost = sin_extreme_affinity_boost

        # Constraints
        self.min_stable_act = min_stable_act
        self.min_stable_agg = min_stable_agg
        self.max_stable_act = max_stable_act
        self.max_stable_agg = max_stable_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with neurogenesis + tagging + homeostatic tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Neurogenesis state: stable and young populations
        act_stable = set(initial_act)
        agg_stable = set(initial_agg)
        act_young: Dict[int, Dict] = {}
        agg_young: Dict[int, Dict] = {}

        # Contribution tracking
        act_contribution = jnp.zeros(NUM_ACTIVATIONS)
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_contribution = act_contribution.at[i].set(0.5)

        agg_contribution = jnp.zeros(NUM_AGGREGATIONS)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_contribution = agg_contribution.at[i].set(0.5)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Neurogenesis
            'act_stable': act_stable,
            'agg_stable': agg_stable,
            'act_young': act_young,
            'agg_young': agg_young,
            'act_contribution': act_contribution,
            'agg_contribution': agg_contribution,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Stats
            'act_total_births': 0,
            'act_total_survivals': 0,
            'act_total_prunings': 0,
            'agg_total_births': 0,
            'agg_total_survivals': 0,
            'agg_total_prunings': 0,
            'capture_events': 0,
            'homeostatic_corrections': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 900000),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _create_mask(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        n_funcs: int,
    ) -> jnp.ndarray:
        """Create mask from stable and young populations."""
        mask = jnp.zeros(n_funcs)
        for i in stable:
            if 0 <= i < n_funcs:
                mask = mask.at[i].set(1.0)
        for i in young.keys():
            if 0 <= i < n_funcs:
                mask = mask.at[i].set(1.0)
        return mask

    def _maybe_birth_neuron(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        captured: jnp.ndarray,
        key: jax.random.PRNGKey,
        generation: int,
        max_young: int,
        n_funcs: int,
        prefer_indices: List[int] = None,
    ) -> Tuple[Set[int], Dict[int, Dict], Optional[int]]:
        """Possibly birth a new neuron with preference for important indices."""
        key1, key2 = jax.random.split(key)
        new_young = dict(young)
        born = None

        if (len(new_young) < max_young and
            jax.random.uniform(key1) < self.neurogenesis_rate):
            # Available = not stable, not young, not already captured
            available = [
                i for i in range(n_funcs)
                if i not in stable and i not in new_young and captured[i] < 0.5
            ]
            if available:
                # Prefer specific indices (sin, extreme aggs)
                if prefer_indices:
                    preferred = [i for i in prefer_indices if i in available]
                    if preferred and jax.random.uniform(jax.random.split(key2)[0]) < 0.6:
                        available = preferred

                idx = int(jax.random.randint(key2, (), 0, len(available)))
                new_func = available[idx]
                new_young[new_func] = {'birth_gen': generation, 'contribution': 0.0}
                born = new_func

        return stable, new_young, born

    def _mature_neurons(
        self,
        stable: Set[int],
        young: Dict[int, Dict],
        contribution: jnp.ndarray,
        tags: jnp.ndarray,
        captured: jnp.ndarray,
        partner_mean_contrib: float,
        generation: int,
        max_stable: int,
    ) -> Tuple[Set[int], Dict[int, Dict], jnp.ndarray, List[int], List[int]]:
        """Process neuron maturation with tag-influenced survival."""
        new_stable = set(stable)
        new_young = {}
        new_captured = captured.copy()
        survived = []
        pruned = []

        for func, info in young.items():
            age = generation - info['birth_gen']
            if age >= self.maturation_period:
                func_contrib = float(contribution[func])
                func_tag = float(tags[func])

                # Cross-domain boost
                effective_threshold = self.survival_threshold - partner_mean_contrib * self.cross_survival_boost

                # Tag-influenced survival (high tag = better chance)
                effective_threshold -= func_tag * 0.1

                if func_contrib >= effective_threshold:
                    if len(new_stable) < max_stable:
                        new_stable.add(func)
                        survived.append(func)
                        # Matured neurons with high tags get captured immediately
                        if func_tag > self.tag_threshold:
                            new_captured = new_captured.at[func].set(1.0)
                    else:
                        pruned.append(func)
                else:
                    pruned.append(func)
            else:
                new_young[func] = info

        return new_stable, new_young, new_captured, survived, pruned

    def _update_contributions(
        self,
        contribution: jnp.ndarray,
        mask: jnp.ndarray,
        young: Dict[int, Dict],
        improved: bool,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, Dict[int, Dict]]:
        """Update contribution tracking with young plasticity."""
        new_contribution = contribution * self.contribution_decay
        new_young = {}

        for i in range(n_funcs):
            if mask[i] > 0.5:
                current = float(new_contribution[i])
                if improved:
                    boost = self.contribution_boost
                    # Young neurons are more plastic
                    if i in young:
                        boost *= self.young_plasticity
                    new_contribution = new_contribution.at[i].set(current + boost)
                else:
                    new_contribution = new_contribution.at[i].set(current + 0.01)

        for func, info in young.items():
            new_info = dict(info)
            new_info['contribution'] = float(new_contribution[func])
            new_young[func] = new_info

        return jnp.clip(new_contribution, 0, 2.0), new_young

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_young: Dict[int, Dict],
        agg_young: Dict[int, Dict],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with young neuron boost."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                # Young neurons tag more easily
                if i in act_young:
                    tag_strength *= self.young_tag_boost
                # Sin boost
                if i == 4:
                    tag_strength *= self.extreme_tag_boost
                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                # Young neurons tag more easily
                if j in agg_young:
                    tag_strength *= self.young_tag_boost
                # Extreme boost
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost
                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        act_young: Dict[int, Dict],
        agg_young: Dict[int, Dict],
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Attempt capture - but NOT for young neurons (must mature first)."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, 0

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                # Capture activations (excluding young)
                for i in range(NUM_ACTIVATIONS):
                    if (hist_act_tags[i] > self.tag_threshold and
                        new_act_captured[i] < 0.5 and
                        i not in act_young):  # NOT young
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1

                # Capture aggregations (excluding young)
                for j in range(NUM_AGGREGATIONS):
                    if (hist_agg_tags[j] > self.agg_tag_threshold and
                        new_agg_captured[j] < 0.5 and
                        j not in agg_young):  # NOT young
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1

        return new_act_captured, new_agg_captured, capture_count

    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute extreme/averaging ratio for homeostatic balance."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    def _update_cross_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        improvement: float,
    ) -> jnp.ndarray:
        """Update cross-domain affinity matrix."""
        new_affinity = cross_affinity.copy()

        if improvement > 0:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    for j in range(NUM_AGGREGATIONS):
                        if agg_mask[j] > 0.5:
                            current = cross_affinity[i, j]
                            boost = self.cross_learning_rate * improvement
                            if i == 4 and j in CORE_EXTREME_AGGS:
                                boost *= (1 + self.sin_extreme_affinity_boost)
                            new_affinity = new_affinity.at[i, j].set(
                                min(1.0, current + boost)
                            )

        return new_affinity

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with combined neurogenesis + tag + homeostatic mechanisms."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        improvement = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === CONTRIBUTION UPDATE (neurogenesis) ===
        new_act_contrib, act_young = self._update_contributions(
            state['act_contribution'], state['act_mask'],
            state['act_young'], improved, NUM_ACTIVATIONS
        )
        new_agg_contrib, agg_young = self._update_contributions(
            state['agg_contribution'], state['agg_mask'],
            state['agg_young'], improved, NUM_AGGREGATIONS
        )

        # === TAGGING (with young boost) ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags'],
            act_young, agg_young
        )

        # Update tag history
        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # Cross-domain mean contributions for survival boost
        act_active = mask_to_indices(state['act_mask'])
        agg_active = mask_to_indices(state['agg_mask'])
        act_mean_contrib = float(jnp.mean(jnp.array([new_act_contrib[i] for i in act_active]))) if act_active else 0
        agg_mean_contrib = float(jnp.mean(jnp.array([new_agg_contrib[i] for i in agg_active]))) if agg_active else 0

        # === MATURATION (neurogenesis with tag influence) ===
        act_stable, act_young, new_act_captured, act_survived, act_pruned = self._mature_neurons(
            set(state['act_stable']), act_young, new_act_contrib,
            new_act_tags, state['act_captured'],
            agg_mean_contrib, generation, self.max_stable_act
        )
        agg_stable, agg_young, new_agg_captured, agg_survived, agg_pruned = self._mature_neurons(
            set(state['agg_stable']), agg_young, new_agg_contrib,
            new_agg_tags, state['agg_captured'],
            act_mean_contrib, generation, self.max_stable_agg
        )

        # === CAPTURE (tag-and-capture, excluding young) ===
        new_act_captured, new_agg_captured, capture_count = self._attempt_capture(
            new_act_tags, new_agg_tags,
            new_act_captured, new_agg_captured,
            act_young, agg_young,
            new_tag_history, generation, improved
        )

        # === BIRTH (neurogenesis with preference for sin/extreme) ===
        act_stable, act_young, act_born = self._maybe_birth_neuron(
            act_stable, act_young, new_act_captured, k1, generation,
            self.max_young_act, NUM_ACTIVATIONS, prefer_indices=[4]  # Prefer sin
        )
        agg_stable, agg_young, agg_born = self._maybe_birth_neuron(
            agg_stable, agg_young, new_agg_captured, k2, generation,
            self.max_young_agg, NUM_AGGREGATIONS, prefer_indices=list(CORE_EXTREME_AGGS)
        )

        # === CREATE MASKS ===
        new_act_mask = self._create_mask(act_stable, act_young, NUM_ACTIVATIONS)
        new_agg_mask = self._create_mask(agg_stable, agg_young, NUM_AGGREGATIONS)

        # === CROSS-DOMAIN AFFINITY ===
        new_cross_affinity = self._update_cross_affinity(
            state['cross_affinity'], state['act_mask'], state['agg_mask'], improvement
        )

        # === HOMEOSTATIC TRACKING ===
        extreme_ratio = self._compute_extreme_ratio(new_agg_mask)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_stable': act_stable,
            'agg_stable': agg_stable,
            'act_young': act_young,
            'agg_young': agg_young,
            'act_contribution': new_act_contrib,
            'agg_contribution': new_agg_contrib,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'cross_affinity': new_cross_affinity,
            'act_total_births': state['act_total_births'] + (1 if act_born else 0),
            'act_total_survivals': state['act_total_survivals'] + len(act_survived),
            'act_total_prunings': state['act_total_prunings'] + len(act_pruned),
            'agg_total_births': state['agg_total_births'] + (1 if agg_born else 0),
            'agg_total_survivals': state['agg_total_survivals'] + len(agg_survived),
            'agg_total_prunings': state['agg_total_prunings'] + len(agg_pruned),
            'capture_events': state['capture_events'] + capture_count,
            'homeostatic_corrections': state['homeostatic_corrections'],
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Neurogenesis metrics
            'act_n_stable': len(act_stable),
            'act_n_young': len(act_young),
            'agg_n_stable': len(agg_stable),
            'agg_n_young': len(agg_young),
            'act_born': act_born,
            'agg_born': agg_born,
            'act_survived': act_survived,
            'agg_survived': agg_survived,
            'act_pruned': act_pruned,
            'agg_pruned': agg_pruned,
            'act_survival_rate': (new_state['act_total_survivals'] / max(new_state['act_total_births'], 1)) * 100,
            'agg_survival_rate': (new_state['agg_total_survivals'] / max(new_state['agg_total_births'], 1)) * 100,
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'sin_is_young': 4 in act_young,
            'sin_is_stable': 4 in act_stable,
            'capture_events': new_state['capture_events'],
            # Homeostatic metrics
            'extreme_ratio': extreme_ratio,
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with neurogenesis + tag + homeostatic status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            # Neurogenesis
            'act_n_stable': len(state['act_stable']),
            'act_n_young': len(state['act_young']),
            'agg_n_stable': len(state['agg_stable']),
            'agg_n_young': len(state['agg_young']),
            'act_total_births': state['act_total_births'],
            'agg_total_births': state['agg_total_births'],
            # Tagging
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'sin_is_young': 4 in state['act_young'],
            'sin_is_stable': 4 in state['act_stable'],
            'capture_events': state['capture_events'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
