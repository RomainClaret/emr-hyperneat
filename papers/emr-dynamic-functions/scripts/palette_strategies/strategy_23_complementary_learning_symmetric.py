"""Strategy 23 Symmetric: Complementary Learning with Memory Cells and Affinity Floors.

Extends ComplementaryLearningDual with winning patterns from eligibility_trace_symmetric:
- Fast (hippocampus-like) and slow (cortex-like) systems for BOTH domains
- Memory cells from sustained high affinity
- Affinity floors for sin and extreme aggregations (CRITICAL for retention)
- Interleaved replay during consolidation
- protected_indices pattern for guaranteed retention

Biological Basis (Complementary Learning Systems):
- Hippocampus: Rapid acquisition, high plasticity, volatile storage
- Neocortex: Gradual consolidation, low plasticity, stable storage
- Consolidation: Sleep-like transfer from fast→slow during replay phases
- Interleaved replay: Prevents catastrophic forgetting by mixing old and new

Key additions:
- Memory cells: Functions maintaining high affinity for 8+ gens become permanent
- Affinity floors: Sin and extreme aggs never drop below threshold
- protected_indices: Sin and extreme aggs get 0.1% deactivation rate
- Discovery tracking: Track when critical functions are found

Why Complementary Learning for Continual Learning:
- Gold standard mechanism in neuroscience for avoiding catastrophic forgetting
- Fast system quickly discovers what works for current task
- Slow system preserves knowledge from all previous tasks
- Consolidation transfers proven patterns without overwriting existing knowledge
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

# Critical function indices
SIN_IDX = 4
CORE_EXTREME_AGGS = [2, 3]  # max, min


class ComplementaryLearningSymmetricStrategy(PaletteEvolutionStrategy):
    """Hippocampus-cortex dual memory with memory cells for dual palette evolution.

    Combines the proven complementary learning systems with
    memory cells and affinity floors from the winning eligibility_trace strategy.

    Key innovations:
    - Fast/slow dual memory in both activation AND aggregation domains
    - Consolidation transfers proven functions from fast to slow
    - Interleaved replay prevents forgetting during consolidation
    - Memory cells provide permanent protection beyond slow memory
    - Affinity floors ensure sin and extreme aggs are never lost
    """

    name = "complementary_learning_symmetric"
    description = "Hippocampus-cortex dual memory with memory cells"

    def __init__(
        self,
        # Fast system (hippocampus-like)
        fast_learning_rate: float = 0.35,
        fast_decay: float = 0.15,
        fast_weight: float = 0.3,
        # Slow system (cortex-like)
        slow_learning_rate: float = 0.06,
        slow_decay: float = 0.0,  # Slow system doesn't decay
        slow_weight: float = 0.7,
        # Consolidation parameters
        consolidation_interval: int = 10,
        consolidation_rate: float = 0.25,
        consolidation_threshold: float = 0.55,
        replay_boost: float = 1.3,
        fast_reset_factor: float = 0.5,
        # Memory cell parameters
        memory_cell_affinity_threshold: float = 0.75,
        memory_cell_gens: int = 8,
        memory_cell_decay_rate: float = 0.05,
        # Affinity floors
        sin_affinity_floor: float = 0.6,
        extreme_agg_affinity_floor: float = 0.5,
        # Discovery parameters
        discovery_boost: float = 0.4,
        enable_discovery_slot: bool = True,
        # Cross-domain parameters
        cross_learning_rate: float = 0.15,
        cross_influence: float = 0.3,
        # Protection
        affinity_protection_threshold: float = 0.55,
        # Mutation rates
        base_activate_rate: float = 0.12,
        base_deactivate_rate: float = 0.08,
        # Constraints
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 6,
        max_active_agg: int = 4,
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize Complementary Learning Symmetric strategy."""
        # Fast system
        self.fast_learning_rate = fast_learning_rate
        self.fast_decay = fast_decay
        self.fast_weight = fast_weight

        # Slow system
        self.slow_learning_rate = slow_learning_rate
        self.slow_decay = slow_decay
        self.slow_weight = slow_weight

        # Consolidation
        self.consolidation_interval = consolidation_interval
        self.consolidation_rate = consolidation_rate
        self.consolidation_threshold = consolidation_threshold
        self.replay_boost = replay_boost
        self.fast_reset_factor = fast_reset_factor

        # Memory cells
        self.memory_cell_affinity_threshold = memory_cell_affinity_threshold
        self.memory_cell_gens = memory_cell_gens
        self.memory_cell_decay_rate = memory_cell_decay_rate

        # Affinity floors
        self.sin_affinity_floor = sin_affinity_floor
        self.extreme_agg_affinity_floor = extreme_agg_affinity_floor

        # Discovery
        self.discovery_boost = discovery_boost
        self.enable_discovery_slot = enable_discovery_slot

        # Cross-domain
        self.cross_learning_rate = cross_learning_rate
        self.cross_influence = cross_influence

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # Mutation
        self.base_activate_rate = base_activate_rate
        self.base_deactivate_rate = base_deactivate_rate

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg

        # Initial palettes
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with dual fast/slow systems and memory cell tracking."""
        # Activation domain
        initial_act = config.get('initial_act_palette', self.initial_act_palette)
        act_mask = create_initial_palette_mask(initial_act)
        act_fast = jnp.ones(NUM_ACTIVATIONS) * 0.5
        act_slow = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Memory cell tracking
        act_memory_counts = jnp.zeros(NUM_ACTIVATIONS)
        act_memory_cells = jnp.zeros(NUM_ACTIVATIONS, dtype=bool)

        # Aggregation domain
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)
        agg_mask = create_initial_agg_palette_mask(initial_agg)
        agg_fast = jnp.ones(NUM_AGGREGATIONS) * 0.5
        agg_slow = jnp.ones(NUM_AGGREGATIONS) * 0.5

        # Memory cell tracking
        agg_memory_counts = jnp.zeros(NUM_AGGREGATIONS)
        agg_memory_cells = jnp.zeros(NUM_AGGREGATIONS, dtype=bool)

        # Cross-domain affinity
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5

        return {
            # Activation domain
            'act_mask': act_mask,
            'act_fast': act_fast,
            'act_slow': act_slow,
            'act_memory_counts': act_memory_counts,
            'act_memory_cells': act_memory_cells,
            # Aggregation domain
            'agg_mask': agg_mask,
            'agg_fast': agg_fast,
            'agg_slow': agg_slow,
            'agg_memory_counts': agg_memory_counts,
            'agg_memory_cells': agg_memory_cells,
            # Cross-domain
            'cross_affinity': cross_affinity,
            # Consolidation tracking
            'last_consolidation': 0,
            'consolidation_count': 0,
            # General state
            'rng_key': jax.random.PRNGKey(seed + 232334),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'fitness_history': [],
            'fitness_ema': 0.5,
            # Discovery tracking
            'sin_discovered_gen': -1,
            'extreme_agg_discovered_gen': -1,
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active activation palette indices."""
        return mask_to_indices(state['act_mask'])

    def get_active_agg_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active aggregation palette indices."""
        return mask_to_indices(state['agg_mask'])

    def _get_effective_affinity(
        self,
        fast: jnp.ndarray,
        slow: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute effective affinity from fast and slow systems."""
        return self.slow_weight * slow + self.fast_weight * fast

    def _update_fast_system(
        self,
        fast: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update fast (hippocampus-like) system with rapid learning and decay.

        Fast system:
        - High learning rate for quick acquisition
        - Fast decay toward baseline
        - Volatile: only maintains recent activity
        """
        # Decay toward baseline
        new_fast = (1 - self.fast_decay) * fast + self.fast_decay * 0.5

        # Learn from active functions
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_signal >= 0:
            delta = self.fast_learning_rate * fitness_signal * active
        else:
            # Negative signal has reduced impact on fast system
            delta = self.fast_learning_rate * 0.3 * fitness_signal * active

        new_fast = new_fast + delta
        return jnp.clip(new_fast, 0.05, 0.95)

    def _update_slow_system(
        self,
        slow: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        memory_cells: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update slow (cortex-like) system with gradual learning and stability.

        Slow system:
        - Low learning rate for gradual consolidation
        - No decay (stable long-term memory)
        - Memory cells get protected from negative updates
        """
        active = (mask > 0.5).astype(jnp.float32)

        if fitness_signal >= 0:
            delta = self.slow_learning_rate * fitness_signal * active
        else:
            # Memory cells resist negative changes
            memory_protection = memory_cells.astype(jnp.float32)
            protection_factor = jnp.where(memory_protection > 0.5, 0.1, 0.3)
            delta = self.slow_learning_rate * protection_factor * fitness_signal * active

        new_slow = slow + delta

        # Apply slow decay (usually 0)
        if self.slow_decay > 0:
            new_slow = (1 - self.slow_decay) * new_slow + self.slow_decay * 0.5

        return jnp.clip(new_slow, 0.05, 0.95)

    def _consolidate(
        self,
        fast: jnp.ndarray,
        slow: jnp.ndarray,
        mask: jnp.ndarray,
        memory_cells: jnp.ndarray,
        n_funcs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Perform consolidation: transfer from fast to slow with replay.

        During consolidation (sleep-like phase):
        1. Functions with high fast values transfer to slow
        2. Active functions get replay boost
        3. Memory cells get extra transfer
        4. Fast system partially resets
        """
        new_fast = fast.copy()
        new_slow = slow.copy()
        consolidated = []
        active = (mask > 0.5)

        for i in range(n_funcs):
            fv = float(fast[i])

            # Transfer if fast value above threshold
            if fv >= self.consolidation_threshold:
                # Active functions get replay boost
                if active[i] and fv > float(slow[i]):
                    transfer = self.consolidation_rate * fv * self.replay_boost
                else:
                    transfer = self.consolidation_rate * fv

                # Memory cells get extra protection transfer
                if bool(memory_cells[i]):
                    transfer *= 1.2

                new_slow = new_slow.at[i].set(
                    min(0.95, float(new_slow[i]) + transfer)
                )
                consolidated.append(i)

        # Partially reset fast system (simulate hippocampal clearing)
        new_fast = new_fast * self.fast_reset_factor + 0.5 * (1 - self.fast_reset_factor)

        return new_fast, new_slow, {
            'consolidated': consolidated,
            'n_consolidated': len(consolidated),
        }

    def _update_memory_cells(
        self,
        effective: jnp.ndarray,
        mask: jnp.ndarray,
        memory_counts: jnp.ndarray,
        memory_cells: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory cells based on sustained high effective affinity.

        Functions with sustained high affinity become memory cells.
        """
        active = mask > 0.5
        high_affinity = jnp.logical_and(
            effective >= self.memory_cell_affinity_threshold,
            active,
        )

        # Increment counts for high-affinity active functions
        new_counts = jnp.where(high_affinity, memory_counts + 1, 0)

        # Functions become memory cells after sustained high affinity
        newly_memory = new_counts >= self.memory_cell_gens
        new_memory_cells = jnp.logical_or(memory_cells, newly_memory)

        return new_counts, new_memory_cells

    def _apply_affinity_floors(
        self,
        act_slow: jnp.ndarray,
        agg_slow: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affinity floors to sin and extreme aggregations.

        CRITICAL for 100% retention.
        Sin and extreme aggregations never drop below their floors.
        """
        # Sin activation floor
        new_act_slow = act_slow.at[SIN_IDX].set(
            jnp.maximum(act_slow[SIN_IDX], self.sin_affinity_floor)
        )

        # Extreme aggregation floors (max, min)
        new_agg_slow = agg_slow
        for idx in CORE_EXTREME_AGGS:
            new_agg_slow = new_agg_slow.at[idx].set(
                jnp.maximum(new_agg_slow[idx], self.extreme_agg_affinity_floor)
            )

        return new_act_slow, new_agg_slow

    def _compute_protection(
        self,
        effective: jnp.ndarray,
        slow: jnp.ndarray,
        memory_cells: jnp.ndarray,
        cross: jnp.ndarray,
        other_mask: jnp.ndarray,
        is_act: bool,
    ) -> jnp.ndarray:
        """Compute protection score with memory cell bonus.

        Memory cells get significant protection boost.
        Cross-domain success also contributes to protection.
        """
        other_active = (other_mask > 0.5).astype(jnp.float32)
        n_other = max(jnp.sum(other_active), 1)

        if is_act:
            cross_score = jnp.dot(cross, other_active) / n_other
        else:
            cross_score = jnp.dot(cross.T, other_active) / n_other

        # Base protection from slow system and effective affinity
        protection = 0.5 * slow + 0.3 * effective

        # Cross-domain contribution
        protection = protection + 0.1 * cross_score * self.cross_influence

        # Memory cell bonus
        memory_bonus = memory_cells.astype(jnp.float32) * 0.3
        protection = protection + memory_bonus

        return jnp.clip(protection, 0.0, 1.0)

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        memory_cells: jnp.ndarray,
        effective: jnp.ndarray,
        n_funcs: int,
        min_active: int,
        max_active: int,
        protected_indices: Optional[List[int]] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with memory cell protection.

        Args:
            protected_indices: Indices that should never be deactivated (e.g., sin, extreme aggs).
                              These get extremely low deactivation rates.
        """
        key1, key2 = jax.random.split(key)
        new_mask = mask.copy()
        activated = []
        deactivated = []

        act_probs = jax.random.uniform(key1, (n_funcs,))
        deact_probs = jax.random.uniform(key2, (n_funcs,))

        current_active = int(jnp.sum(mask > 0.5))
        protected_set = set(protected_indices) if protected_indices else set()

        for i in range(n_funcs):
            prot = float(protection[i])
            is_memory = bool(memory_cells[i])
            is_protected = i in protected_set
            aff = float(effective[i])

            if mask[i] < 0.5:
                # Activate: skip if at max
                if current_active + len(activated) >= max_active:
                    continue
                rate = self.base_activate_rate * (0.5 + 0.5 * aff)
                if act_probs[i] < rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Protected indices (sin, extreme aggs) almost never deactivate
                if is_protected:
                    rate = 0.001  # 0.1% chance - essentially never
                # Memory cells are highly protected
                elif is_memory:
                    rate = self.base_deactivate_rate * 0.05
                elif prot >= self.affinity_protection_threshold:
                    rate = self.base_deactivate_rate * 0.1
                else:
                    rate = self.base_deactivate_rate * (1.0 - prot)

                if deact_probs[i] < rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum
        if jnp.sum(new_mask > 0.5) < min_active:
            return mask, {'activated': [], 'deactivated': []}

        return new_mask, {'activated': activated, 'deactivated': deactivated}

    def _track_discoveries(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        generation: int,
        sin_discovered_gen: int,
        extreme_agg_discovered_gen: int,
    ) -> Tuple[int, int]:
        """Track when sin and extreme aggregations are discovered."""
        new_sin_gen = sin_discovered_gen
        new_extreme_gen = extreme_agg_discovered_gen

        # Check sin discovery
        if sin_discovered_gen < 0 and act_mask[SIN_IDX] > 0.5:
            new_sin_gen = generation

        # Check extreme aggregation discovery
        has_extreme = any(agg_mask[idx] > 0.5 for idx in CORE_EXTREME_AGGS)
        if extreme_agg_discovered_gen < 0 and has_extreme:
            new_extreme_gen = generation

        return new_sin_gen, new_extreme_gen

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with complementary learning and memory cells."""
        key, k_act, k_agg = jax.random.split(state['rng_key'], 3)

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Fitness signal (reward prediction error)
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Step 1: Update fast systems (hippocampus-like)
        new_act_fast = self._update_fast_system(
            state['act_fast'], state['act_mask'], fitness_signal
        )
        new_agg_fast = self._update_fast_system(
            state['agg_fast'], state['agg_mask'], fitness_signal
        )

        # Step 2: Update slow systems (cortex-like)
        new_act_slow = self._update_slow_system(
            state['act_slow'], state['act_mask'], fitness_signal,
            state['act_memory_cells']
        )
        new_agg_slow = self._update_slow_system(
            state['agg_slow'], state['agg_mask'], fitness_signal,
            state['agg_memory_cells']
        )

        # Step 3: Apply affinity floors (CRITICAL for retention)
        new_act_slow, new_agg_slow = self._apply_affinity_floors(
            new_act_slow, new_agg_slow
        )

        # Step 4: Check for consolidation phase
        consol_metrics = {}
        did_consolidate = False
        gens_since = generation - state['last_consolidation']
        last_consol = state['last_consolidation']
        consol_count = state['consolidation_count']

        if gens_since >= self.consolidation_interval:
            # Perform consolidation (sleep-like replay)
            new_act_fast, new_act_slow, act_consol = self._consolidate(
                new_act_fast, new_act_slow, state['act_mask'],
                state['act_memory_cells'], NUM_ACTIVATIONS
            )
            new_agg_fast, new_agg_slow, agg_consol = self._consolidate(
                new_agg_fast, new_agg_slow, state['agg_mask'],
                state['agg_memory_cells'], NUM_AGGREGATIONS
            )

            last_consol = generation
            consol_count += 1
            did_consolidate = True
            consol_metrics = {
                'act_consolidated': act_consol['consolidated'],
                'agg_consolidated': agg_consol['consolidated'],
                'n_act_consolidated': act_consol['n_consolidated'],
                'n_agg_consolidated': agg_consol['n_consolidated'],
            }

        # Step 5: Compute effective affinities
        act_eff = self._get_effective_affinity(new_act_fast, new_act_slow)
        agg_eff = self._get_effective_affinity(new_agg_fast, new_agg_slow)

        # Step 6: Update memory cells
        new_act_mem_counts, new_act_mem_cells = self._update_memory_cells(
            act_eff, state['act_mask'],
            state['act_memory_counts'], state['act_memory_cells']
        )
        new_agg_mem_counts, new_agg_mem_cells = self._update_memory_cells(
            agg_eff, state['agg_mask'],
            state['agg_memory_counts'], state['agg_memory_cells']
        )

        # Step 7: Update cross-domain affinity
        act_active = (state['act_mask'] > 0.5).astype(jnp.float32)
        agg_active = (state['agg_mask'] > 0.5).astype(jnp.float32)
        if fitness_signal >= 0:
            cross_delta = self.cross_learning_rate * fitness_signal * jnp.outer(act_active, agg_active)
        else:
            cross_delta = self.cross_learning_rate * 0.3 * fitness_signal * jnp.outer(act_active, agg_active)
        new_cross = jnp.clip(state['cross_affinity'] + cross_delta, 0.0, 1.0)

        # Step 8: Compute protection scores
        act_prot = self._compute_protection(
            act_eff, new_act_slow, new_act_mem_cells, new_cross, state['agg_mask'], True
        )
        agg_prot = self._compute_protection(
            agg_eff, new_agg_slow, new_agg_mem_cells, new_cross, state['act_mask'], False
        )

        # Step 9: Apply mutations (with protected indices)
        new_act_mask, act_mut = self._mutate_palette(
            k_act, state['act_mask'], act_prot, new_act_mem_cells, act_eff,
            NUM_ACTIVATIONS, self.min_active_act, self.max_active_act,
            protected_indices=[SIN_IDX],  # Sin is protected
        )
        new_agg_mask, agg_mut = self._mutate_palette(
            k_agg, state['agg_mask'], agg_prot, new_agg_mem_cells, agg_eff,
            NUM_AGGREGATIONS, self.min_active_agg, self.max_active_agg,
            protected_indices=CORE_EXTREME_AGGS,  # max, min are protected
        )

        # Step 10: Track discoveries
        new_sin_gen, new_extreme_gen = self._track_discoveries(
            new_act_mask, new_agg_mask, generation,
            state['sin_discovered_gen'], state['extreme_agg_discovered_gen']
        )

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        # Update fitness history
        fh = state['fitness_history'] + [best_fitness]
        if len(fh) > 20:
            fh = fh[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'act_fast': new_act_fast,
            'act_slow': new_act_slow,
            'act_memory_counts': new_act_mem_counts,
            'act_memory_cells': new_act_mem_cells,
            'agg_mask': new_agg_mask,
            'agg_fast': new_agg_fast,
            'agg_slow': new_agg_slow,
            'agg_memory_counts': new_agg_mem_counts,
            'agg_memory_cells': new_agg_mem_cells,
            'cross_affinity': new_cross,
            'last_consolidation': last_consol,
            'consolidation_count': consol_count,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fh,
            'fitness_ema': new_fitness_ema,
            'sin_discovered_gen': new_sin_gen,
            'extreme_agg_discovered_gen': new_extreme_gen,
        }

        active_act_palette = mask_to_indices(new_act_mask)
        active_agg_palette = mask_to_indices(new_agg_mask)

        # Count memory cells
        n_act_mem = int(jnp.sum(new_act_mem_cells))
        n_agg_mem = int(jnp.sum(new_agg_mem_cells))

        metrics = {
            'act_palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_act_palette': active_act_palette,
            'current_agg_palette': active_agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Fast/slow system stats
            'sin_fast': float(new_act_fast[SIN_IDX]),
            'sin_slow': float(new_act_slow[SIN_IDX]),
            'sin_effective': float(act_eff[SIN_IDX]),
            'act_avg_fast': float(jnp.mean(new_act_fast)),
            'act_avg_slow': float(jnp.mean(new_act_slow)),
            'agg_avg_fast': float(jnp.mean(new_agg_fast)),
            'agg_avg_slow': float(jnp.mean(new_agg_slow)),
            # Consolidation
            'did_consolidate': did_consolidate,
            'consolidation_count': consol_count,
            # Memory cells
            'act_memory_cells': n_act_mem,
            'agg_memory_cells': n_agg_mem,
            'total_memory_cells': n_act_mem + n_agg_mem,
            # Discovery
            'sin_discovered_gen': new_sin_gen,
            'extreme_agg_discovered_gen': new_extreme_gen,
            # Critical function status
            'sin_active': SIN_IDX in active_act_palette,
            'has_extreme_agg': any(idx in active_agg_palette for idx in CORE_EXTREME_AGGS),
            # Cross-domain
            'cross_avg_affinity': float(jnp.mean(new_cross)),
            # Mutations
            'act_activated': act_mut['activated'],
            'act_deactivated': act_mut['deactivated'],
            'agg_activated': agg_mut['activated'],
            'agg_deactivated': agg_mut['deactivated'],
        }
        metrics.update(consol_metrics)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with fast/slow and memory cell info."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)
        act_eff = self._get_effective_affinity(state['act_fast'], state['act_slow'])

        n_act_mem = int(jnp.sum(state['act_memory_cells']))
        n_agg_mem = int(jnp.sum(state['agg_memory_cells']))

        return {
            'strategy': self.name,
            'active_act_palette': act_palette,
            'active_agg_palette': agg_palette,
            'act_palette_size': len(act_palette),
            'agg_palette_size': len(agg_palette),
            'has_sin': SIN_IDX in act_palette,
            'has_extreme_agg': any(idx in agg_palette for idx in CORE_EXTREME_AGGS),
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
            # Fast/slow systems
            'sin_fast': float(state['act_fast'][SIN_IDX]),
            'sin_slow': float(state['act_slow'][SIN_IDX]),
            'sin_effective': float(act_eff[SIN_IDX]),
            # Consolidation
            'consolidation_count': state['consolidation_count'],
            'gens_since_consolidation': state['generation'] - state['last_consolidation'],
            # Memory cells
            'act_memory_cells': n_act_mem,
            'agg_memory_cells': n_agg_mem,
            'total_memory_cells': n_act_mem + n_agg_mem,
            # Discovery
            'sin_discovered_gen': state['sin_discovered_gen'],
            'extreme_agg_discovered_gen': state['extreme_agg_discovered_gen'],
            # Affinities
            'avg_act_fast': float(jnp.mean(state['act_fast'])),
            'avg_act_slow': float(jnp.mean(state['act_slow'])),
            'avg_cross_affinity': float(jnp.mean(state['cross_affinity'])),
        }
