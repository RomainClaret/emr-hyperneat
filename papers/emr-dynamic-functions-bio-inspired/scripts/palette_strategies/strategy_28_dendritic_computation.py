"""Strategy 28: Dendritic Computation (Zone-Based Local Processing).

Implements dendritic compartmentalization - local computation in spatial
neighborhoods before global integration. Functions are grouped into dendritic
zones that learn and compete independently.

Biological Basis:
- Dendrites compute locally before signals reach soma
- Different dendritic regions perform distinct filtering
- Local nonlinearities within dendritic compartments
- Integration at soma combines compartment outputs

Key Insight:
- Previous strategies assume global function interactions
- This can cause interference: useful functions cancel each other
- Dendritic computation groups functions into zones
- Only coherent function groups compete, reducing interference

Zone Mechanism:
    # Functions assigned to zones based on similarity
    zones = {
        'oscillatory': [4, 11, 12, 13, 15],  # sin, burst, resonator, etc.
        'monotonic': [0, 1, 2, 3],            # identity, tanh, sigmoid, relu
        'spatial': [7, 14, 16, 17],           # gaussian, locality
        'nonlinear': [5, 6, 8, 9, 10]         # step, leaky_relu, etc.
    }

    # Zone-local learning
    for zone z:
        if functions in zone z co-activate with fitness improvement:
            zone_memory[z] += learning_rate

    # Zone competition
    active_zone = argmax(zone_fitness_contributions)
    palette = functions from active_zone + exploration from others

Expected improvements:
- Reduces interference between dissimilar functions
- Finds coherent function groups faster
- Better exploration of zone-specific combinations
- Natural organization by functional type
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


class DendriticComputationStrategy(PaletteEvolutionStrategy):
    """Zone-based local processing before global integration.

    Functions are grouped into dendritic zones. Learning happens locally
    within zones, and zones compete for palette representation. This
    reduces interference between dissimilar function types.
    """

    name = "dendritic_computation"
    description = "Zone-based local computation with compartmentalized learning"

    # Default zone assignments (functional grouping)
    DEFAULT_ZONES = {
        0: 1,   # identity -> monotonic
        1: 1,   # tanh -> monotonic
        2: 1,   # sigmoid -> monotonic
        3: 1,   # relu -> monotonic
        4: 0,   # sin -> oscillatory
        5: 3,   # step -> nonlinear
        6: 3,   # leaky_relu -> nonlinear
        7: 2,   # gaussian -> spatial
        8: 3,   # softplus -> nonlinear
        9: 3,   # elu -> nonlinear
        10: 3,  # swish -> nonlinear
        11: 0,  # burst -> oscillatory
        12: 0,  # resonator -> oscillatory
        13: 0,  # osc_adapt -> oscillatory
        14: 2,  # locality -> spatial
        15: 0,  # receptive -> oscillatory
        16: 2,  # spatial_decay -> spatial
        17: 2,  # edge_detector -> spatial
    }

    ZONE_NAMES = ['oscillatory', 'monotonic', 'spatial', 'nonlinear']

    def __init__(
        self,
        # Zone parameters
        n_zones: int = 4,
        zone_assignments: Dict[int, int] = None,
        # Integration mode
        integration_mode: str = "weighted_sum",  # or "winner_take_all"
        winner_slots: int = 3,  # Functions from winning zone
        exploration_slots: int = 2,  # Functions from other zones
        # Zone learning
        zone_learning_rate: float = 0.15,
        zone_decay: float = 0.92,
        zone_threshold: float = 0.3,  # Activation threshold for zone
        # Function learning within zones
        function_learning_rate: float = 0.1,
        function_decay: float = 0.95,
        # Mutation rates
        zone_activate_rate: float = 0.15,
        zone_deactivate_rate: float = 0.08,
        cross_zone_activate_rate: float = 0.03,  # Low for exploration
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Dendritic Computation strategy.

        Args:
            n_zones: Number of dendritic zones
            zone_assignments: Map function index -> zone index
            integration_mode: How to combine zones (weighted_sum or winner_take_all)
            winner_slots: Functions from winning zone (for winner_take_all)
            exploration_slots: Functions from non-winning zones (exploration)
            zone_learning_rate: How fast zones learn
            zone_decay: Zone memory persistence
            zone_threshold: Threshold for zone activation
            function_learning_rate: Learning within zones
            function_decay: Function memory within zones
        """
        # Zone config
        self.n_zones = n_zones
        self.zone_assignments = zone_assignments or self.DEFAULT_ZONES
        self.integration_mode = integration_mode
        self.winner_slots = winner_slots
        self.exploration_slots = exploration_slots

        # Learning
        self.zone_learning_rate = zone_learning_rate
        self.zone_decay = zone_decay
        self.zone_threshold = zone_threshold
        self.function_learning_rate = function_learning_rate
        self.function_decay = function_decay

        # Mutation
        self.zone_activate_rate = zone_activate_rate
        self.zone_deactivate_rate = zone_deactivate_rate
        self.cross_zone_activate_rate = cross_zone_activate_rate

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

        # Build zone index arrays
        self._build_zone_indices()

    def _build_zone_indices(self):
        """Build arrays mapping functions to zones."""
        self.function_to_zone = jnp.zeros(NUM_ACTIVATIONS, dtype=jnp.int32)
        for func_idx, zone_idx in self.zone_assignments.items():
            if func_idx < NUM_ACTIVATIONS:
                self.function_to_zone = self.function_to_zone.at[func_idx].set(zone_idx)

        # Build zone member lists
        self.zone_members = {z: [] for z in range(self.n_zones)}
        for func_idx, zone_idx in self.zone_assignments.items():
            if func_idx < NUM_ACTIVATIONS and zone_idx < self.n_zones:
                self.zone_members[zone_idx].append(func_idx)

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with zone memories."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Zone-level memories
        zone_memories = jnp.zeros(self.n_zones)
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                zone_idx = int(self.function_to_zone[i])
                zone_memories = zone_memories.at[zone_idx].add(0.2)

        # Per-function memories within zones [n_zones, NUM_ACTIVATIONS]
        function_memories = jnp.zeros((self.n_zones, NUM_ACTIVATIONS))
        for i in initial:
            if 0 <= i < NUM_ACTIVATIONS:
                zone_idx = int(self.function_to_zone[i])
                function_memories = function_memories.at[zone_idx, i].set(0.3)

        # Zone activation history
        zone_activations = jnp.zeros(self.n_zones)

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 282828),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Zone state
            'zone_memories': zone_memories,
            'function_memories': function_memories,
            'zone_activations': zone_activations,
            'dominant_zone': -1,  # Currently most active zone
            # Tracking
            'previous_mask': mask,
            'fitness_history': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _compute_zone_contributions(
        self,
        mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
    ) -> jnp.ndarray:
        """Compute fitness contribution per zone.

        Zones with active functions that correlate with improvement get credit.
        """
        improvement = max(0, fitness - prev_fitness)
        zone_contributions = jnp.zeros(self.n_zones)

        for zone_idx in range(self.n_zones):
            # Count active functions in this zone
            zone_active = 0.0
            for func_idx in self.zone_members.get(zone_idx, []):
                if func_idx < NUM_ACTIVATIONS and mask[func_idx] > 0.5:
                    zone_active += 1.0

            if zone_active > 0:
                # Zone gets credit proportional to its representation
                zone_contributions = zone_contributions.at[zone_idx].set(
                    improvement * zone_active / max(jnp.sum(mask > 0.5), 1.0)
                )

        return zone_contributions

    def _update_zone_memories(
        self,
        zone_memories: jnp.ndarray,
        zone_contributions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update zone memories with new contributions."""
        # Decay existing memories
        new_memories = self.zone_decay * zone_memories

        # Add new contributions
        new_memories = new_memories + self.zone_learning_rate * zone_contributions

        return jnp.clip(new_memories, 0.0, 1.0)

    def _update_function_memories(
        self,
        function_memories: jnp.ndarray,
        mask: jnp.ndarray,
        fitness: float,
        prev_fitness: float,
    ) -> jnp.ndarray:
        """Update per-function memories within zones."""
        new_memories = self.function_decay * function_memories
        improvement = max(0, fitness - prev_fitness)

        for zone_idx in range(self.n_zones):
            for func_idx in self.zone_members.get(zone_idx, []):
                if func_idx < NUM_ACTIVATIONS and mask[func_idx] > 0.5:
                    # Active function in this zone gets credit
                    new_memories = new_memories.at[zone_idx, func_idx].add(
                        self.function_learning_rate * improvement
                    )

        return jnp.clip(new_memories, 0.0, 1.0)

    def _select_dominant_zone(
        self,
        zone_memories: jnp.ndarray,
    ) -> int:
        """Select the currently dominant zone."""
        return int(jnp.argmax(zone_memories))

    def _apply_dendritic_mutation(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        zone_memories: jnp.ndarray,
        function_memories: jnp.ndarray,
        dominant_zone: int,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with zone-based rates."""
        key1, key2 = jax.random.split(key)
        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        new_mask = mask.copy()
        activated = []
        deactivated = []
        zone_changes = {z: {'in': 0, 'out': 0} for z in range(self.n_zones)}

        for func_idx in range(NUM_ACTIVATIONS):
            zone_idx = int(self.function_to_zone[func_idx])
            func_memory = float(function_memories[zone_idx, func_idx])
            zone_memory = float(zone_memories[zone_idx])
            is_dominant = (zone_idx == dominant_zone)

            if mask[func_idx] < 0.5:
                # Inactive: maybe activate
                if is_dominant:
                    # Dominant zone: higher activation rate
                    rate = self.zone_activate_rate * (1.0 + zone_memory + func_memory)
                else:
                    # Non-dominant: cross-zone exploration
                    rate = self.cross_zone_activate_rate * (1.0 + func_memory)

                if activate_probs[func_idx] < rate:
                    new_mask = new_mask.at[func_idx].set(1.0)
                    activated.append(func_idx)
                    zone_changes[zone_idx]['in'] += 1

            else:
                # Active: maybe deactivate
                if is_dominant:
                    # Dominant zone: lower deactivation (protected)
                    rate = self.zone_deactivate_rate * (1.0 - zone_memory * 0.5)
                else:
                    # Non-dominant: higher deactivation
                    rate = self.zone_deactivate_rate * (1.5 - func_memory)

                if deactivate_probs[func_idx] < rate:
                    new_mask = new_mask.at[func_idx].set(0.0)
                    deactivated.append(func_idx)
                    zone_changes[zone_idx]['out'] += 1

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'zone_changes': zone_changes,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with dendritic zone dynamics."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Step 1: Compute zone contributions
        zone_contributions = self._compute_zone_contributions(
            state['mask'],
            best_fitness,
            prev_best_fitness,
        )

        # Step 2: Update zone memories
        new_zone_memories = self._update_zone_memories(
            state['zone_memories'],
            zone_contributions,
        )

        # Step 3: Update function memories within zones
        new_function_memories = self._update_function_memories(
            state['function_memories'],
            state['mask'],
            best_fitness,
            prev_best_fitness,
        )

        # Step 4: Select dominant zone
        dominant_zone = self._select_dominant_zone(new_zone_memories)

        # Step 5: Apply dendritic mutation
        new_mask, mutation_info = self._apply_dendritic_mutation(
            subkey,
            state['mask'],
            new_zone_memories,
            new_function_memories,
            dominant_zone,
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Track fitness
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Zone state
            'zone_memories': new_zone_memories,
            'function_memories': new_function_memories,
            'zone_activations': zone_contributions,
            'dominant_zone': dominant_zone,
            # Tracking
            'previous_mask': state['mask'],
            'fitness_history': fitness_history,
        }

        # Compute metrics
        active_palette = mask_to_indices(new_mask)

        # Zone representation in current palette
        zone_representation = {z: 0 for z in range(self.n_zones)}
        for func_idx in active_palette:
            zone_idx = int(self.function_to_zone[func_idx])
            zone_representation[zone_idx] += 1

        # Top functions per zone
        top_per_zone = {}
        for zone_idx in range(self.n_zones):
            zone_funcs = [(i, float(new_function_memories[zone_idx, i]))
                          for i in self.zone_members.get(zone_idx, [])]
            zone_funcs.sort(key=lambda x: -x[1])
            top_per_zone[zone_idx] = zone_funcs[:3]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Zone info
            'dominant_zone': dominant_zone,
            'dominant_zone_name': self.ZONE_NAMES[dominant_zone] if dominant_zone < len(self.ZONE_NAMES) else f'zone_{dominant_zone}',
            'zone_memories': [float(m) for m in new_zone_memories],
            'zone_representation': zone_representation,
            # Sin status
            'has_sin': 4 in active_palette,
            'sin_zone': int(self.function_to_zone[4]),
            'sin_memory': float(new_function_memories[int(self.function_to_zone[4]), 4]),
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with zone status."""
        palette = self.get_active_palette(state)
        zone_memories = state['zone_memories']
        function_memories = state['function_memories']

        # Zone representation
        zone_rep = {z: [] for z in range(self.n_zones)}
        for func_idx in palette:
            zone_idx = int(self.function_to_zone[func_idx])
            zone_rep[zone_idx].append(func_idx)

        # Top zone
        dominant = int(jnp.argmax(zone_memories))

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Zone status
            'dominant_zone': dominant,
            'dominant_zone_name': self.ZONE_NAMES[dominant] if dominant < len(self.ZONE_NAMES) else f'zone_{dominant}',
            'zone_memories': [float(m) for m in zone_memories],
            'zone_representation': {self.ZONE_NAMES[z]: zone_rep[z] for z in range(self.n_zones)},
            # Sin-specific
            'sin_zone_memory': float(zone_memories[int(self.function_to_zone[4])]),
        }
