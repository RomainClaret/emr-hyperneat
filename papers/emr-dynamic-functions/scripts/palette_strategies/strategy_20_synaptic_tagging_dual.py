"""Strategy 20 Dual: Synaptic Tagging for Both Activation AND Aggregation.

Extends SynapticTagging to jointly evolve both activation and aggregation palettes
with two-stage tag-and-capture learning in both domains.

Key mechanisms extended to dual:
1. Separate tag systems for activations and aggregations
2. Capture events affect both domains simultaneously
3. Cross-domain tags track successful act-agg combinations
4. Captured functions in either domain get permanent protection
5. Max_active constraints prevent antagonism (max_act=6, max_agg=4)

Biological rationale:
- Synaptic tagging operates at all synapses, not just one modality
- Global reward signals (dopamine) trigger capture across domains
- Cross-modal associations can be tagged and captured together

Expected improvement:
- Fewer false positives in BOTH domains
- Cross-domain capture links successful act-agg combinations
- More robust to noise in both domains
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    create_initial_agg_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    NUM_AGGREGATIONS,
    DEFAULT_PALETTE_INDICES,
    DEFAULT_AGG_PALETTE_INDICES,
)


class SynapticTaggingDualStrategy(PaletteEvolutionStrategy):
    """Synaptic tagging with dual palette evolution.

    Two-stage learning in both activation and aggregation domains.
    """

    name = "synaptic_tagging_dual"
    description = "Dual palette tagging and capture mechanism"

    def __init__(
        self,
        # Tag parameters
        tag_threshold: float = 0.15,
        tag_strength_per_event: float = 0.3,
        max_tag_strength: float = 1.0,
        tag_decay_rate: float = 0.15,
        tag_min_threshold: float = 0.1,
        # Capture parameters
        capture_window: int = 5,
        capture_threshold: float = 0.30,
        capture_efficiency: float = 0.5,
        capture_bonus: float = 0.2,
        captured_protection: float = 0.8,
        captured_affinity_min: float = 0.7,
        # Cross-domain
        cross_learning_rate: float = 0.12,
        cross_influence: float = 0.25,
        # Base learning
        affinity_learning_rate: float = 0.10,
        mutation_rate: float = 0.15,
        affinity_protection_threshold: float = 0.55,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.tag_threshold = tag_threshold
        self.tag_strength_per_event = tag_strength_per_event
        self.max_tag_strength = max_tag_strength
        self.tag_decay_rate = tag_decay_rate
        self.tag_min_threshold = tag_min_threshold
        self.capture_window = capture_window
        self.capture_threshold = capture_threshold
        self.capture_efficiency = capture_efficiency
        self.capture_bonus = capture_bonus
        self.captured_protection = captured_protection
        self.captured_affinity_min = captured_affinity_min
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence
        self.affinity_learning_rate = affinity_learning_rate
        self.mutation_rate = mutation_rate
        self.affinity_protection_threshold = affinity_protection_threshold
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        act_mask = create_initial_palette_mask(config.get('initial_act_palette', self.initial_act_palette))
        agg_mask = create_initial_agg_palette_mask(config.get('initial_agg_palette', self.initial_agg_palette))

        return {
            'act_mask': act_mask,
            'act_affinity': jnp.ones(NUM_ACTIVATIONS) * 0.5,
            'act_tags': jnp.zeros(NUM_ACTIVATIONS),
            'act_tag_gen': jnp.ones(NUM_ACTIVATIONS) * -100,
            'act_captured': set(),
            'agg_mask': agg_mask,
            'agg_affinity': jnp.ones(NUM_AGGREGATIONS) * 0.5,
            'agg_tags': jnp.zeros(NUM_AGGREGATIONS),
            'agg_tag_gen': jnp.ones(NUM_AGGREGATIONS) * -100,
            'agg_captured': set(),
            'cross_affinity': jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5,
            'rng_key': jax.random.PRNGKey(seed + 202021),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            'fitness_ema': 0.5,
            'act_tag_events': 0,
            'agg_tag_events': 0,
            'act_capture_events': 0,
            'agg_capture_events': 0,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['agg_mask'])

    def _create_tags(
        self, tags: jnp.ndarray, tag_gen: jnp.ndarray, mask: jnp.ndarray,
        improvement: float, generation: int, n_funcs: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        if improvement < self.tag_threshold:
            return tags, tag_gen, 0
        new_tags, new_gen = tags.copy(), tag_gen.copy()
        active = (mask > 0.5).astype(jnp.float32)
        n_tagged = 0
        for i in range(n_funcs):
            if float(active[i]) > 0.5:
                inc = self.tag_strength_per_event * (improvement / max(self.tag_threshold, 0.01))
                new_tags = new_tags.at[i].set(min(self.max_tag_strength, float(tags[i]) + inc))
                new_gen = new_gen.at[i].set(float(generation))
                n_tagged += 1
        return new_tags, new_gen, n_tagged

    def _decay_tags(self, tags: jnp.ndarray, tag_gen: jnp.ndarray, gen: int, n_funcs: int) -> jnp.ndarray:
        new_tags = tags.copy()
        for i in range(n_funcs):
            if float(tags[i]) < self.tag_min_threshold:
                continue
            age = gen - int(tag_gen[i])
            eff_decay = self.tag_decay_rate * (1.0 + 0.1 * age)
            new_str = float(tags[i]) * (1.0 - eff_decay)
            new_tags = new_tags.at[i].set(new_str if new_str >= self.tag_min_threshold else 0.0)
        return new_tags

    def _attempt_capture(
        self, tags: jnp.ndarray, tag_gen: jnp.ndarray, affinity: jnp.ndarray,
        captured: Set[int], mask: jnp.ndarray, improvement: float, gen: int, n_funcs: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Set[int], List[int]]:
        if improvement < self.capture_threshold:
            return tags, affinity, captured, []
        new_tags, new_aff = tags.copy(), affinity.copy()
        new_captured = captured.copy()
        newly = []
        for i in range(n_funcs):
            if i in captured:
                continue
            strength, age = float(tags[i]), gen - int(tag_gen[i])
            if strength >= self.tag_min_threshold and age <= self.capture_window:
                boost = strength * self.capture_efficiency + self.capture_bonus
                new_aff = new_aff.at[i].set(max(self.captured_affinity_min, float(affinity[i]) + boost))
                new_captured.add(i)
                newly.append(i)
                new_tags = new_tags.at[i].set(0.0)
        return new_tags, new_aff, new_captured, newly

    def _update_affinity(
        self, affinity: jnp.ndarray, mask: jnp.ndarray, captured: Set[int], fs: float, n_funcs: int
    ) -> jnp.ndarray:
        new_aff = affinity.copy()
        active = (mask > 0.5).astype(jnp.float32)
        for i in range(n_funcs):
            if i in captured:
                if fs > 0 and float(active[i]) > 0.5:
                    delta = self.affinity_learning_rate * 0.3 * fs
                    new_aff = new_aff.at[i].set(min(0.95, float(new_aff[i]) + delta))
                if float(new_aff[i]) < self.captured_affinity_min:
                    new_aff = new_aff.at[i].set(self.captured_affinity_min)
            else:
                if float(active[i]) > 0.5:
                    delta = self.affinity_learning_rate * fs if fs >= 0 else self.affinity_learning_rate * 0.3 * fs
                    new_aff = new_aff.at[i].set(max(0.05, min(0.95, float(new_aff[i]) + delta)))
        return new_aff

    def _compute_protection(
        self, affinity: jnp.ndarray, tags: jnp.ndarray, captured: Set[int],
        cross: jnp.ndarray, other_mask: jnp.ndarray, is_act: bool, n_funcs: int
    ) -> jnp.ndarray:
        prot = affinity.copy()
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)
        if is_act:
            cross_score = jnp.dot(cross, other_active) / n_other
        else:
            cross_score = jnp.dot(cross.T, other_active) / n_other
        for i in range(n_funcs):
            if i in captured:
                prot = prot.at[i].set(self.captured_protection)
            else:
                tag_boost = float(tags[i]) * 0.2
                prot = prot.at[i].set(min(0.95, float(prot[i]) + tag_boost + 0.15 * float(cross_score[i]) * self.cross_influence))
        return prot

    def _mutate_palette(
        self, key: jax.random.PRNGKey, mask: jnp.ndarray, protection: jnp.ndarray,
        captured: Set[int], n_funcs: int, min_active: int, max_active: int, is_act: bool
    ) -> Tuple[jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated, deactivated = [], []
        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))
        current = int(jnp.sum(mask > 0.5))

        for i in range(n_funcs):
            prot = float(protection[i])
            is_cap = i in captured
            if mask[i] < 0.5:
                if current + len(activated) >= max_active:
                    continue
                eff = self.mutation_rate * 2.0 if is_cap else self.mutation_rate * (0.5 + prot)
                if act_probs[i] < eff:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if is_cap:
                    dr = self.mutation_rate * 0.02
                elif prot >= self.affinity_protection_threshold:
                    dr = self.mutation_rate * 0.1
                else:
                    dr = self.mutation_rate * (1.0 - prot)
                if deact_probs[i] < dr:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            return mask, {'activated': [], 'deactivated': []}

        prefix = 'act_' if is_act else 'agg_'
        return new_mask, {f'{prefix}activated': activated, f'{prefix}deactivated': deactivated}

    def post_generation_update(
        self, state: Dict[str, Any], generation: int, best_fitness: float,
        prev_best_fitness: float, population_data: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        alpha = 0.2
        new_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        local_imp = best_fitness - prev_best_fitness
        global_imp = best_fitness - state['fitness_ema']
        fs = max(-1.0, min(1.0, (best_fitness - new_ema) / max(0.1, new_ema)))

        # Decay tags
        act_tags = self._decay_tags(state['act_tags'], state['act_tag_gen'], generation, NUM_ACTIVATIONS)
        agg_tags = self._decay_tags(state['agg_tags'], state['agg_tag_gen'], generation, NUM_AGGREGATIONS)

        # Create tags
        act_tags, act_tag_gen, n_act_tagged = self._create_tags(act_tags, state['act_tag_gen'], state['act_mask'], local_imp, generation, NUM_ACTIVATIONS)
        agg_tags, agg_tag_gen, n_agg_tagged = self._create_tags(agg_tags, state['agg_tag_gen'], state['agg_mask'], local_imp, generation, NUM_AGGREGATIONS)

        # Capture
        act_tags, act_aff, act_cap, act_newly = self._attempt_capture(act_tags, act_tag_gen, state['act_affinity'], state['act_captured'], state['act_mask'], global_imp, generation, NUM_ACTIVATIONS)
        agg_tags, agg_aff, agg_cap, agg_newly = self._attempt_capture(agg_tags, agg_tag_gen, state['agg_affinity'], state['agg_captured'], state['agg_mask'], global_imp, generation, NUM_AGGREGATIONS)

        # Update affinities
        act_aff = self._update_affinity(act_aff, state['act_mask'], act_cap, fs, NUM_ACTIVATIONS)
        agg_aff = self._update_affinity(agg_aff, state['agg_mask'], agg_cap, fs, NUM_AGGREGATIONS)

        # Cross-domain
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        cross_delta = self.cross_learning_rate * fs * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Protection
        act_prot = self._compute_protection(act_aff, act_tags, act_cap, new_cross, state['agg_mask'], True, NUM_ACTIVATIONS)
        agg_prot = self._compute_protection(agg_aff, agg_tags, agg_cap, new_cross, state['act_mask'], False, NUM_AGGREGATIONS)

        # Mutations
        new_act_mask, act_mut = self._mutate_palette(k_act, state['act_mask'], act_prot, act_cap, NUM_ACTIVATIONS, self.min_active_act, self.max_active_act, True)
        new_agg_mask, agg_mut = self._mutate_palette(k_agg, state['agg_mask'], agg_prot, agg_cap, NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg, False)

        fh = state['fitness_history'] + [best_fitness]
        if len(fh) > 20:
            fh = fh[-20:]

        new_state = {
            'act_mask': new_act_mask, 'act_affinity': act_aff, 'act_tags': act_tags, 'act_tag_gen': act_tag_gen, 'act_captured': act_cap,
            'agg_mask': new_agg_mask, 'agg_affinity': agg_aff, 'agg_tags': agg_tags, 'agg_tag_gen': agg_tag_gen, 'agg_captured': agg_cap,
            'cross_affinity': new_cross, 'rng_key': key, 'generation': generation + 1, 'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best, 'strategy_name': self.name, 'fitness_history': fh, 'fitness_ema': new_ema,
            'act_tag_events': state['act_tag_events'] + n_act_tagged, 'agg_tag_events': state['agg_tag_events'] + n_agg_tagged,
            'act_capture_events': state['act_capture_events'] + len(act_newly), 'agg_capture_events': state['agg_capture_events'] + len(agg_newly),
        }

        metrics = {
            'act_palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_act_palette': mask_to_indices(new_act_mask),
            'current_agg_palette': mask_to_indices(new_agg_mask),
            'sin_affinity': float(act_aff[4]) if 4 < len(act_aff) else 0.0,
            'sin_captured': 4 in act_cap,
            'act_newly_captured': act_newly, 'agg_newly_captured': agg_newly,
            'act_n_captured': len(act_cap), 'agg_n_captured': len(agg_cap),
        }
        metrics.update(act_mut)
        metrics.update(agg_mut)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'strategy': self.name,
            'active_act_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'sin_captured': 4 in state['act_captured'],
            'generation': state['generation'],
            'act_n_captured': len(state['act_captured']),
            'agg_n_captured': len(state['agg_captured']),
        }
