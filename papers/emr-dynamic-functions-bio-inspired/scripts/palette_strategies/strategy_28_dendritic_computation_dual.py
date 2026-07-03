"""Strategy 28D: Dendritic Computation Dual (Zone-Based for Both Palettes).

Extends Dendritic Computation to jointly evolve activation AND aggregation
function palettes with zone-based local processing for both domains.

Key mechanisms:
1. Functions grouped into zones by functional similarity
2. Zone-local learning before global integration
3. Dominant zone protection, cross-zone exploration
4. Cross-domain: Zones can span both activation and aggregation

Biological basis:
- Dendrites compute locally before signals reach soma
- Different dendritic regions perform distinct filtering
- Compartmentalized learning reduces interference
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .base_strategy import (
    PaletteEvolutionStrategy,
    create_initial_palette_mask,
    mask_to_indices,
    NUM_ACTIVATIONS,
    DEFAULT_PALETTE_INDICES,
)

NUM_AGGREGATIONS = 6
DEFAULT_AGGREGATION_INDICES = [0, 1]


def create_initial_agg_mask(indices: List[int]) -> jnp.ndarray:
    mask = jnp.zeros(NUM_AGGREGATIONS)
    for idx in indices:
        if 0 <= idx < NUM_AGGREGATIONS:
            mask = mask.at[idx].set(1.0)
    return mask


def agg_mask_to_indices(mask: jnp.ndarray) -> List[int]:
    return [int(i) for i in range(NUM_AGGREGATIONS) if mask[i] > 0.5]


class DendriticComputationDualStrategy(PaletteEvolutionStrategy):
    """Zone-based local processing for both activation and aggregation palettes."""

    name = "dendritic_computation_dual"
    description = "Zone-based dendritic computation for dual palette evolution"

    # Activation zone assignments
    ACT_ZONES = {
        0: 1, 1: 1, 2: 1, 3: 1,     # monotonic
        4: 0, 11: 0, 12: 0, 13: 0, 15: 0,  # oscillatory
        5: 3, 6: 3, 8: 3, 9: 3, 10: 3,    # nonlinear
        7: 2, 14: 2, 16: 2, 17: 2,        # spatial
    }

    # Aggregation zone assignments
    AGG_ZONES = {
        0: 0, 1: 0,  # additive (sum, mean)
        2: 1, 3: 1,  # extremal (max, min)
        4: 2, 5: 2,  # multiplicative (product, maxabs)
    }

    ACT_ZONE_NAMES = ['oscillatory', 'monotonic', 'spatial', 'nonlinear']
    AGG_ZONE_NAMES = ['additive', 'extremal', 'multiplicative']

    def __init__(
        self,
        # Zone config
        n_act_zones: int = 4,
        n_agg_zones: int = 3,
        # Zone learning
        zone_learning_rate: float = 0.15,
        zone_decay: float = 0.92,
        # Function learning
        function_learning_rate: float = 0.1,
        function_decay: float = 0.95,
        # Mutation rates
        zone_activate_rate: float = 0.15,
        zone_deactivate_rate: float = 0.08,
        cross_zone_activate_rate: float = 0.03,
        # Cross-domain
        cross_domain_learning_rate: float = 0.08,
        # Constraints
        max_active_act: int = 6,
        max_active_agg: int = 4,
        min_active_act: int = 2,
        min_active_agg: int = 1,
        # General
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        self.n_act_zones = n_act_zones
        self.n_agg_zones = n_agg_zones

        self.zone_learning_rate = zone_learning_rate
        self.zone_decay = zone_decay
        self.function_learning_rate = function_learning_rate
        self.function_decay = function_decay

        self.zone_activate_rate = zone_activate_rate
        self.zone_deactivate_rate = zone_deactivate_rate
        self.cross_zone_activate_rate = cross_zone_activate_rate

        self.cross_domain_learning_rate = cross_domain_learning_rate

        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg

        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGGREGATION_INDICES

        self._build_zone_indices()

    def _build_zone_indices(self):
        """Build zone lookups for both domains."""
        # Activation zones
        self.act_func_to_zone = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        for i, z in self.ACT_ZONES.items():
            if i < NUM_ACTIVATIONS:
                self.act_func_to_zone = self.act_func_to_zone.at[i].set(z)

        self.act_zone_members = {z: [] for z in range(self.n_act_zones)}
        for i, z in self.ACT_ZONES.items():
            if i < NUM_ACTIVATIONS and z < self.n_act_zones:
                self.act_zone_members[z].append(i)

        # Aggregation zones
        self.agg_func_to_zone = jnp.zeros(NUM_AGGREGATIONS, dtype=jnp.int32)
        for i, z in self.AGG_ZONES.items():
            if i < NUM_AGGREGATIONS:
                self.agg_func_to_zone = self.agg_func_to_zone.at[i].set(z)

        self.agg_zone_members = {z: [] for z in range(self.n_agg_zones)}
        for i, z in self.AGG_ZONES.items():
            if i < NUM_AGGREGATIONS and z < self.n_agg_zones:
                self.agg_zone_members[z].append(i)

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_mask(initial_agg)

        # Zone memories
        act_zone_memories = jnp.zeros(self.n_act_zones)
        agg_zone_memories = jnp.zeros(self.n_agg_zones)

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                z = int(self.act_func_to_zone[i])
                act_zone_memories = act_zone_memories.at[z].add(0.2)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                z = int(self.agg_func_to_zone[i])
                agg_zone_memories = agg_zone_memories.at[z].add(0.2)

        # Function memories within zones
        act_func_memories = jnp.zeros((self.n_act_zones, NUM_ACTIVATIONS))
        agg_func_memories = jnp.zeros((self.n_agg_zones, NUM_AGGREGATIONS))

        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                z = int(self.act_func_to_zone[i])
                act_func_memories = act_func_memories.at[z, i].set(0.3)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                z = int(self.agg_func_to_zone[i])
                agg_func_memories = agg_func_memories.at[z, i].set(0.3)

        # Cross-domain zone affinity [act_zones x agg_zones]
        cross_zone_affinity = jnp.ones((self.n_act_zones, self.n_agg_zones)) * 0.5

        return {
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            'act_zone_memories': act_zone_memories,
            'agg_zone_memories': agg_zone_memories,
            'act_func_memories': act_func_memories,
            'agg_func_memories': agg_func_memories,
            'cross_zone_affinity': cross_zone_affinity,
            'act_dominant_zone': -1,
            'agg_dominant_zone': -1,
            'rng_key': jax.random.PRNGKey(seed + 282828),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'mask': act_mask,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        return agg_mask_to_indices(state['agg_mask'])

    def _compute_zone_contributions(
        self,
        mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
        func_to_zone: jnp.ndarray,
        zone_members: Dict,
        n_zones: int,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Compute fitness contribution per zone."""
        improvement = max(0, fitness - prev_fitness)
        contributions = jnp.zeros(n_zones)

        for z in range(n_zones):
            zone_active = 0.0
            for i in zone_members.get(z, []):
                if i < n_funcs and mask[i] > 0.5:
                    zone_active += 1.0

            if zone_active > 0:
                contributions = contributions.at[z].set(
                    improvement * zone_active / max(jnp.sum(mask > 0.5), 1.0)
                )

        return contributions

    def _update_zone_memories(
        self,
        zone_memories: jnp.ndarray,
        contributions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update zone memories."""
        new_memories = self.zone_decay * zone_memories
        new_memories = new_memories + self.zone_learning_rate * contributions
        return jnp.clip(new_memories, 0.0, 1.0)

    def _update_function_memories(
        self,
        func_memories: jnp.ndarray,
        mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
        func_to_zone: jnp.ndarray,
        zone_members: Dict,
        n_zones: int,
        n_funcs: int,
    ) -> jnp.ndarray:
        """Update function memories within zones."""
        new_memories = self.function_decay * func_memories
        improvement = max(0, fitness - prev_fitness)

        for z in range(n_zones):
            for i in zone_members.get(z, []):
                if i < n_funcs and mask[i] > 0.5:
                    new_memories = new_memories.at[z, i].add(
                        self.function_learning_rate * improvement
                    )

        return jnp.clip(new_memories, 0.0, 1.0)

    def _update_cross_zone_affinity(
        self,
        cross_affinity: jnp.ndarray,
        act_zone_memories: jnp.ndarray,
        agg_zone_memories: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        fitness_improved: bool,
    ) -> jnp.ndarray:
        """Update cross-domain zone affinity."""
        # Compute active zone patterns
        act_zone_active = jnp.zeros(self.n_act_zones)
        agg_zone_active = jnp.zeros(self.n_agg_zones)

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                z = int(self.act_func_to_zone[i])
                act_zone_active = act_zone_active.at[z].add(1.0)

        for i in range(NUM_AGGREGATIONS):
            if agg_mask[i] > 0.5:
                z = int(self.agg_func_to_zone[i])
                agg_zone_active = agg_zone_active.at[z].add(1.0)

        # Normalize
        act_zone_active = act_zone_active / max(jnp.sum(act_zone_active), 1.0)
        agg_zone_active = agg_zone_active / max(jnp.sum(agg_zone_active), 1.0)

        # Outer product for co-occurrence
        zone_co_active = jnp.outer(act_zone_active, agg_zone_active)

        if fitness_improved:
            delta = self.cross_domain_learning_rate * zone_co_active
        else:
            delta = -self.cross_domain_learning_rate * 0.3 * zone_co_active

        return jnp.clip(cross_affinity + delta, 0.0, 1.0)

    def _apply_dendritic_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        zone_memories: jnp.ndarray,
        func_memories: jnp.ndarray,
        dominant_zone: int,
        func_to_zone: jnp.ndarray,
        zone_members: Dict,
        n_zones: int,
        n_funcs: int,
        max_active: int,
        min_active: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with zone-based rates."""
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (n_funcs,))
        deactivate_probs = jax.random.uniform(key2, (n_funcs,))

        new_mask = mask.copy()
        activated = []
        deactivated = []

        for i in range(n_funcs):
            z = int(func_to_zone[i])
            func_mem = float(func_memories[z, i])
            zone_mem = float(zone_memories[z])
            is_dominant = (z == dominant_zone)

            if mask[i] < 0.5:
                current_active = int(jnp.sum(mask > 0.5))
                if current_active >= max_active:
                    continue

                if is_dominant:
                    rate = self.zone_activate_rate * (1.0 + zone_mem + func_mem)
                else:
                    rate = self.cross_zone_activate_rate * (1.0 + func_mem)

                if activate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if is_dominant:
                    rate = self.zone_deactivate_rate * (1.0 - zone_mem * 0.5)
                else:
                    rate = self.zone_deactivate_rate * (1.5 - func_mem)

                if deactivate_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        key, subkey1, subkey2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        new_stagnation = 0 if improved else state['stagnation_count'] + 1
        new_best = best_fitness if improved else state['best_fitness_seen']

        # Compute zone contributions
        act_contributions = self._compute_zone_contributions(
            state['act_mask'], best_fitness, prev_best_fitness,
            self.act_func_to_zone, self.act_zone_members,
            self.n_act_zones, NUM_ACTIVATIONS
        )
        agg_contributions = self._compute_zone_contributions(
            state['agg_mask'], best_fitness, prev_best_fitness,
            self.agg_func_to_zone, self.agg_zone_members,
            self.n_agg_zones, NUM_AGGREGATIONS
        )

        # Update zone memories
        new_act_zone_mem = self._update_zone_memories(state['act_zone_memories'], act_contributions)
        new_agg_zone_mem = self._update_zone_memories(state['agg_zone_memories'], agg_contributions)

        # Update function memories
        new_act_func_mem = self._update_function_memories(
            state['act_func_memories'], state['act_mask'], best_fitness, prev_best_fitness,
            self.act_func_to_zone, self.act_zone_members, self.n_act_zones, NUM_ACTIVATIONS
        )
        new_agg_func_mem = self._update_function_memories(
            state['agg_func_memories'], state['agg_mask'], best_fitness, prev_best_fitness,
            self.agg_func_to_zone, self.agg_zone_members, self.n_agg_zones, NUM_AGGREGATIONS
        )

        # Update cross-domain
        new_cross_zone = self._update_cross_zone_affinity(
            state['cross_zone_affinity'], new_act_zone_mem, new_agg_zone_mem,
            state['act_mask'], state['agg_mask'], improved
        )

        # Select dominant zones
        act_dominant = int(jnp.argmax(new_act_zone_mem))
        agg_dominant = int(jnp.argmax(new_agg_zone_mem))

        # Apply mutations
        new_act_mask, act_mut = self._apply_dendritic_mutation(
            subkey1, state['act_mask'], new_act_zone_mem, new_act_func_mem,
            act_dominant, self.act_func_to_zone, self.act_zone_members,
            self.n_act_zones, NUM_ACTIVATIONS, self.max_active_act, self.min_active_act
        )
        new_agg_mask, agg_mut = self._apply_dendritic_mutation(
            subkey2, state['agg_mask'], new_agg_zone_mem, new_agg_func_mem,
            agg_dominant, self.agg_func_to_zone, self.agg_zone_members,
            self.n_agg_zones, NUM_AGGREGATIONS, self.max_active_agg, self.min_active_agg
        )

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_zone_memories': new_act_zone_mem,
            'agg_zone_memories': new_agg_zone_mem,
            'act_func_memories': new_act_func_mem,
            'agg_func_memories': new_agg_func_mem,
            'cross_zone_affinity': new_cross_zone,
            'act_dominant_zone': act_dominant,
            'agg_dominant_zone': agg_dominant,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'mask': new_act_mask,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = agg_mask_to_indices(new_agg_mask)

        # Zone representation
        act_zone_rep = {z: 0 for z in range(self.n_act_zones)}
        for i in act_palette:
            z = int(self.act_func_to_zone[i])
            act_zone_rep[z] += 1

        agg_zone_rep = {z: 0 for z in range(self.n_agg_zones)}
        for i in agg_palette:
            z = int(self.agg_func_to_zone[i])
            agg_zone_rep[z] += 1

        metrics = {
            'palette_changed': not jnp.allclose(state['act_mask'], new_act_mask),
            'agg_palette_changed': not jnp.allclose(state['agg_mask'], new_agg_mask),
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Zone info
            'act_dominant_zone': act_dominant,
            'agg_dominant_zone': agg_dominant,
            'act_dominant_zone_name': self.ACT_ZONE_NAMES[act_dominant],
            'agg_dominant_zone_name': self.AGG_ZONE_NAMES[agg_dominant],
            'act_zone_memories': [float(m) for m in new_act_zone_mem],
            'agg_zone_memories': [float(m) for m in new_agg_zone_mem],
            'act_zone_representation': act_zone_rep,
            'agg_zone_representation': agg_zone_rep,
            # Cross-domain
            'cross_zone_avg': float(jnp.mean(new_cross_zone)),
            'cross_zone_max': float(jnp.max(new_cross_zone)),
            # Sin status
            'has_sin': 4 in act_palette,
            'sin_zone': int(self.act_func_to_zone[4]),
            'sin_memory': float(new_act_func_mem[int(self.act_func_to_zone[4]), 4]),
            'act_activated': act_mut['activated'],
            'act_deactivated': act_mut['deactivated'],
            'agg_activated': agg_mut['activated'],
            'agg_deactivated': agg_mut['deactivated'],
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'strategy': self.name,
            'active_palette': self.get_active_palette(state),
            'active_agg_palette': self.get_active_agg_palette(state),
            'has_sin': 4 in self.get_active_palette(state),
            'generation': state['generation'],
            'act_dominant_zone': state['act_dominant_zone'],
            'agg_dominant_zone': state['agg_dominant_zone'],
            'act_zone_memories': [float(m) for m in state['act_zone_memories']],
            'agg_zone_memories': [float(m) for m in state['agg_zone_memories']],
            'cross_zone_avg': float(jnp.mean(state['cross_zone_affinity'])),
        }
