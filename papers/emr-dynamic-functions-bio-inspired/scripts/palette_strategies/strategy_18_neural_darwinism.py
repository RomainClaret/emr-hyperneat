"""Strategy 18: Neural Darwinism (Selective Stabilization).

Implements Edelman's Neural Darwinism theory for palette evolution.

Key insight: Standard Hebbian only tracks cooperation (functions that succeed
together). But understanding ANTAGONISM is equally important - which functions
conflict and hurt each other's performance?

Biological Basis (Edelman's Neural Darwinism, 1987):
- Neuronal groups compete for survival
- Groups that cooperate get stabilized
- Groups that conflict get weakened through selective death
- Selection through differential amplification

For palette evolution:
- Track both cooperation AND antagonism between functions
- Cooperation: Functions that succeed together strengthen bonds
- Antagonism: Functions that fail together (or succeed when other removed) conflict
- Selection: Cooperating groups get protected, antagonistic pairs get separated

Key mechanisms:
1. Cooperation matrix: Track pairwise success correlation
2. Antagonism matrix: Track pairwise failure correlation
3. Neuronal group detection: Identify clusters of cooperating functions
4. Selective stabilization: Protect cooperative groups, prune antagonists

Expected improvement over Hebbian:
- Discovers function pairs that CONFLICT
- Avoids keeping antagonistic functions together
- Better function pruning (remove based on conflict, not just inactivity)
- Emergent specialization through group selection
"""

from typing import Dict, Any, List, Optional, Tuple, Set
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


class CriticalPeriodPhase:
    """Critical period developmental phases."""
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    CONSOLIDATION = "consolidation"


class NeuralDarwinismStrategy(PaletteEvolutionStrategy):
    """Neural Darwinism with cooperation/antagonism tracking.

    Implements selective stabilization through group competition.
    """

    name = "neural_darwinism"
    description = "Selective stabilization with cooperation and antagonism matrices"

    def __init__(
        self,
        # Critical period timing
        exploration_end: int = 30,
        confirmation_end: int = 60,
        # Phase rates
        exploration_activate: float = 0.35,
        exploration_deactivate: float = 0.02,
        confirmation_activate: float = 0.10,
        confirmation_deactivate_max: float = 0.15,
        confirmation_deactivate_min: float = 0.01,
        consolidation_activate: float = 0.02,
        consolidation_deactivate: float = 0.01,
        # Neural Darwinism parameters
        cooperation_threshold: float = 0.65,     # Above = cooperative pair
        antagonism_threshold: float = 0.35,      # Below = antagonistic pair
        cooperation_rate: float = 0.15,          # Speed of cooperation learning
        antagonism_rate: float = 0.10,           # Speed of antagonism learning
        selection_pressure: float = 0.20,        # Strength of selection
        group_min_size: int = 2,                 # Min functions to form group
        # Selective death parameters
        antagonism_prune_threshold: float = 0.7, # Antagonism above this triggers pruning
        selective_death_rate: float = 0.1,       # Rate of removing antagonistic functions
        # Base Hebbian parameters
        learning_rate: float = 0.20,
        affinity_protection_threshold: float = 0.55,
        exploration_lr_multiplier: float = 1.5,
        confirmation_lr_multiplier: float = 0.5,
        # Other
        early_consolidation_threshold: float = 0.95,
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Neural Darwinism strategy.

        Args:
            cooperation_threshold: Score above which pairs are cooperative
            antagonism_threshold: Score below which pairs are antagonistic
            cooperation_rate: Learning rate for cooperation matrix
            antagonism_rate: Learning rate for antagonism matrix
            selection_pressure: How strongly selection affects affinity
            group_min_size: Minimum size for neuronal group
            antagonism_prune_threshold: High antagonism triggers removal
            selective_death_rate: Probability of removing antagonistic function
        """
        # Critical period timing
        self.exploration_end = exploration_end
        self.confirmation_end = confirmation_end

        # Phase rates
        self.exploration_activate = exploration_activate
        self.exploration_deactivate = exploration_deactivate
        self.confirmation_activate = confirmation_activate
        self.confirmation_deactivate_max = confirmation_deactivate_max
        self.confirmation_deactivate_min = confirmation_deactivate_min
        self.consolidation_activate = consolidation_activate
        self.consolidation_deactivate = consolidation_deactivate

        # Neural Darwinism parameters
        self.cooperation_threshold = cooperation_threshold
        self.antagonism_threshold = antagonism_threshold
        self.cooperation_rate = cooperation_rate
        self.antagonism_rate = antagonism_rate
        self.selection_pressure = selection_pressure
        self.group_min_size = group_min_size
        self.antagonism_prune_threshold = antagonism_prune_threshold
        self.selective_death_rate = selective_death_rate

        # Hebbian parameters
        self.learning_rate = learning_rate
        self.affinity_protection_threshold = affinity_protection_threshold
        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.confirmation_lr_multiplier = confirmation_lr_multiplier

        # Other
        self.early_consolidation_threshold = early_consolidation_threshold
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def _get_phase(self, generation: int, best_fitness: float) -> str:
        """Determine current phase."""
        if best_fitness >= self.early_consolidation_threshold:
            return CriticalPeriodPhase.CONSOLIDATION

        if generation < self.exploration_end:
            return CriticalPeriodPhase.EXPLORATION
        elif generation < self.confirmation_end:
            return CriticalPeriodPhase.CONFIRMATION
        else:
            return CriticalPeriodPhase.CONSOLIDATION

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with cooperation/antagonism matrices."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Function affinity
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Cooperation matrix: [i,j] = how well i and j work together
        # Starts neutral (0.5)
        cooperation_matrix = jnp.ones((NUM_ACTIVATIONS, NUM_ACTIVATIONS)) * 0.5

        # Antagonism matrix: [i,j] = how much i and j conflict
        # Starts low (no known conflicts)
        antagonism_matrix = jnp.zeros((NUM_ACTIVATIONS, NUM_ACTIVATIONS))

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 181818),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            'phase': CriticalPeriodPhase.EXPLORATION,
            # Core state
            'function_affinity': function_affinity,
            'cooperation_matrix': cooperation_matrix,
            'antagonism_matrix': antagonism_matrix,
            'fitness_history': [],
            'fitness_ema': 0.5,
            # Tracking
            'neuronal_groups': [],           # Detected cooperative groups
            'antagonistic_pairs': [],        # Detected conflicting pairs
            'selection_events': 0,           # Count of selection
            'pruned_by_antagonism': [],      # Functions removed due to conflict
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _update_cooperation_antagonism(
        self,
        cooperation: jnp.ndarray,
        antagonism: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update cooperation and antagonism matrices.

        - Success with i,j active: increase cooperation[i,j]
        - Failure with i,j active: increase antagonism[i,j]
        """
        if phase == CriticalPeriodPhase.EXPLORATION:
            lr = self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr = self.confirmation_lr_multiplier
        else:
            lr = 0.1

        active = (mask > 0.5).astype(jnp.float32)
        co_active = jnp.outer(active, active)

        if fitness_signal > 0:
            # Success: strengthen cooperation among active functions
            coop_delta = self.cooperation_rate * lr * fitness_signal * co_active
            new_cooperation = jnp.clip(cooperation + coop_delta, 0.0, 1.0)

            # Slight reduction in antagonism (success means they work together)
            antag_delta = -self.antagonism_rate * lr * 0.3 * fitness_signal * co_active
            new_antagonism = jnp.clip(antagonism + antag_delta, 0.0, 1.0)

        else:
            # Failure: increase antagonism among active functions
            antag_delta = self.antagonism_rate * lr * abs(fitness_signal) * co_active
            new_antagonism = jnp.clip(antagonism + antag_delta, 0.0, 1.0)

            # Slight reduction in cooperation (failure means they conflict)
            coop_delta = -self.cooperation_rate * lr * 0.3 * abs(fitness_signal) * co_active
            new_cooperation = jnp.clip(cooperation + coop_delta, 0.0, 1.0)

        return new_cooperation, new_antagonism

    def _detect_neuronal_groups(
        self,
        cooperation: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> List[Set[int]]:
        """Detect neuronal groups (clusters of cooperating functions).

        Uses simple threshold-based clustering on cooperation matrix.
        """
        active_indices = [i for i in range(NUM_ACTIVATIONS) if mask[i] > 0.5]

        if len(active_indices) < self.group_min_size:
            return []

        groups = []
        visited = set()

        def find_group(start: int) -> Set[int]:
            """BFS to find connected cooperative functions."""
            group = {start}
            queue = [start]

            while queue:
                current = queue.pop(0)
                for other in active_indices:
                    if other not in group and cooperation[current, other] > self.cooperation_threshold:
                        group.add(other)
                        queue.append(other)

            return group

        for idx in active_indices:
            if idx not in visited:
                group = find_group(idx)
                if len(group) >= self.group_min_size:
                    groups.append(group)
                visited.update(group)

        return groups

    def _detect_antagonistic_pairs(
        self,
        antagonism: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> List[Tuple[int, int]]:
        """Detect pairs of functions with high antagonism."""
        active_indices = [i for i in range(NUM_ACTIVATIONS) if mask[i] > 0.5]
        pairs = []

        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                idx_i = active_indices[i]
                idx_j = active_indices[j]
                if antagonism[idx_i, idx_j] > self.antagonism_prune_threshold:
                    pairs.append((idx_i, idx_j))

        return pairs

    def _apply_selection(
        self,
        affinity: jnp.ndarray,
        cooperation: jnp.ndarray,
        antagonism: jnp.ndarray,
        mask: jnp.ndarray,
        groups: List[Set[int]],
        antagonistic_pairs: List[Tuple[int, int]],
        phase: str,
    ) -> Tuple[jnp.ndarray, List[int]]:
        """Apply selective stabilization.

        - Functions in cooperative groups get affinity boost
        - Functions in antagonistic pairs get affinity penalty
        - Functions with many antagonists may be marked for pruning

        Returns: (new_affinity, functions_to_prune)
        """
        if phase == CriticalPeriodPhase.EXPLORATION:
            pressure = self.selection_pressure * 0.5  # Weaker during exploration
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            pressure = self.selection_pressure * 1.0  # Full during confirmation
        else:
            pressure = self.selection_pressure * 0.3  # Mild during consolidation

        new_affinity = affinity.copy()
        functions_to_prune = []

        # Boost functions in cooperative groups
        for group in groups:
            group_cooperation = 0
            for i in group:
                for j in group:
                    if i != j:
                        group_cooperation += float(cooperation[i, j])

            # Average cooperation within group
            if len(group) > 1:
                avg_cooperation = group_cooperation / (len(group) * (len(group) - 1))

                # Boost proportional to cooperation
                for idx in group:
                    boost = pressure * (avg_cooperation - 0.5)
                    new_affinity = new_affinity.at[idx].set(
                        min(0.95, float(new_affinity[idx]) + boost)
                    )

        # Track antagonism per function
        antagonism_count = {i: 0 for i in range(NUM_ACTIVATIONS)}
        for i, j in antagonistic_pairs:
            antagonism_count[i] += 1
            antagonism_count[j] += 1

            # Both functions in pair get penalty
            penalty = pressure * float(antagonism[i, j])
            new_affinity = new_affinity.at[i].set(
                max(0.05, float(new_affinity[i]) - penalty * 0.5)
            )
            new_affinity = new_affinity.at[j].set(
                max(0.05, float(new_affinity[j]) - penalty * 0.5)
            )

        # Functions with multiple antagonistic relationships may be pruned
        for idx, count in antagonism_count.items():
            if count >= 2:  # Antagonistic with 2+ other functions
                avg_antag = sum(
                    float(antagonism[idx, other])
                    for _, other in antagonistic_pairs if _ == idx
                ) / max(count, 1)

                if avg_antag > self.antagonism_prune_threshold:
                    functions_to_prune.append(idx)

        return new_affinity, functions_to_prune

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_signal: float,
        phase: str,
    ) -> jnp.ndarray:
        """Basic affinity update (before selection)."""
        if phase == CriticalPeriodPhase.EXPLORATION:
            lr = self.learning_rate * self.exploration_lr_multiplier
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            lr = self.learning_rate * self.confirmation_lr_multiplier
        else:
            lr = self.learning_rate * 0.1

        active = (mask > 0.5).astype(jnp.float32)

        if fitness_signal >= 0:
            delta = lr * fitness_signal * active
        else:
            delta = lr * 0.3 * fitness_signal * active  # Anti-learning slower

        return jnp.clip(affinity + delta, 0.0, 1.0)

    def _compute_protection_scores(
        self,
        affinity: jnp.ndarray,
        cooperation: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute protection using cooperation instead of hebbian weights."""
        active = (mask > 0.5).astype(jnp.float32)
        n_active = max(jnp.sum(active), 1)

        pairwise_score = jnp.dot(cooperation, active) / n_active
        protection = 0.7 * affinity + 0.3 * pairwise_score

        return protection

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        phase: str,
        protection_scores: jnp.ndarray,
        functions_to_prune: List[int],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with selective death for antagonistic functions."""
        key1, key2, key3 = jax.random.split(key, 3)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        selectively_killed = []
        protection_info = {}

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))
        death_probs = jax.random.uniform(key3, (NUM_ACTIVATIONS,))

        if phase == CriticalPeriodPhase.EXPLORATION:
            activate_rate = self.exploration_activate
            use_protection = False
            use_selective_death = False
        elif phase == CriticalPeriodPhase.CONFIRMATION:
            activate_rate = self.confirmation_activate
            use_protection = True
            use_selective_death = True
        else:
            activate_rate = self.consolidation_activate
            use_protection = True
            use_selective_death = False  # Too late for pruning

        # First, apply selective death for antagonistic functions
        if use_selective_death:
            for idx in functions_to_prune:
                if mask[idx] > 0.5 and death_probs[idx] < self.selective_death_rate:
                    new_mask = new_mask.at[idx].set(0.0)
                    selectively_killed.append(idx)

        for i in range(NUM_ACTIVATIONS):
            if i in selectively_killed:
                continue  # Already handled

            protection = float(protection_scores[i])

            if mask[i] < 0.5:
                if use_protection and phase == CriticalPeriodPhase.CONFIRMATION:
                    effective_rate = activate_rate * (0.5 + protection)
                else:
                    effective_rate = activate_rate

                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                if phase == CriticalPeriodPhase.CONSOLIDATION:
                    if protection >= self.affinity_protection_threshold:
                        protection_info[i] = f"protected (affinity={protection:.2f})"
                        continue
                    deact_rate = self.consolidation_deactivate

                elif phase == CriticalPeriodPhase.CONFIRMATION:
                    if protection >= self.affinity_protection_threshold:
                        deact_rate = self.confirmation_deactivate_min
                        protection_info[i] = f"protected (affinity={protection:.2f})"
                    else:
                        t = protection / self.affinity_protection_threshold
                        deact_rate = (
                            self.confirmation_deactivate_max * (1 - t) +
                            self.confirmation_deactivate_min * t
                        )
                        protection_info[i] = f"vulnerable (affinity={protection:.2f})"
                else:
                    deact_rate = self.exploration_deactivate

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []
            selectively_killed = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
            'selectively_killed': selectively_killed,
            'protection_info': protection_info,
        }

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with Neural Darwinism selection."""
        key, subkey = jax.random.split(state['rng_key'])

        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        phase = self._get_phase(generation, new_best)
        phase_changed = phase != state['phase']

        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness

        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Step 1: Update cooperation and antagonism matrices
        new_cooperation, new_antagonism = self._update_cooperation_antagonism(
            state['cooperation_matrix'],
            state['antagonism_matrix'],
            state['mask'],
            fitness_signal,
            phase,
        )

        # Step 2: Basic affinity update
        new_affinity = self._update_affinity(
            state['function_affinity'],
            state['mask'],
            fitness_signal,
            phase,
        )

        # Step 3: Detect neuronal groups and antagonistic pairs
        groups = self._detect_neuronal_groups(new_cooperation, state['mask'])
        antagonistic_pairs = self._detect_antagonistic_pairs(new_antagonism, state['mask'])

        # Step 4: Apply selection
        new_affinity, functions_to_prune = self._apply_selection(
            new_affinity,
            new_cooperation,
            new_antagonism,
            state['mask'],
            groups,
            antagonistic_pairs,
            phase,
        )

        # Compute protection scores
        protection_scores = self._compute_protection_scores(
            new_affinity, new_cooperation, state['mask']
        )

        # Apply mutation with selective death
        new_mask, mutation_info = self._mutate_palette(
            subkey, state['mask'], phase, protection_scores, functions_to_prune
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        selection_events = state['selection_events'] + (1 if groups or antagonistic_pairs else 0)
        pruned = state['pruned_by_antagonism'] + mutation_info.get('selectively_killed', [])

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'phase': phase,
            'function_affinity': new_affinity,
            'cooperation_matrix': new_cooperation,
            'antagonism_matrix': new_antagonism,
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            'neuronal_groups': [list(g) for g in groups],
            'antagonistic_pairs': antagonistic_pairs,
            'selection_events': selection_events,
            'pruned_by_antagonism': pruned,
        }

        active_palette = mask_to_indices(new_mask)
        protected_functions = [
            i for i in active_palette
            if protection_scores[i] >= self.affinity_protection_threshold
        ]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            'phase': phase,
            'phase_changed': phase_changed,
            'fitness_signal': fitness_signal,
            'avg_affinity': float(jnp.mean(new_affinity)),
            'max_affinity': float(jnp.max(new_affinity)),
            'sin_affinity': float(new_affinity[4]),
            'n_protected': len(protected_functions),
            'protected_functions': protected_functions,
            # Neural Darwinism stats
            'n_neuronal_groups': len(groups),
            'group_sizes': [len(g) for g in groups],
            'n_antagonistic_pairs': len(antagonistic_pairs),
            'antagonistic_pairs': antagonistic_pairs,
            'selection_events': selection_events,
            'selectively_killed': mutation_info.get('selectively_killed', []),
            'total_pruned': len(pruned),
            'avg_cooperation': float(jnp.mean(new_cooperation)),
            'avg_antagonism': float(jnp.mean(new_antagonism)),
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with cooperation/antagonism stats."""
        palette = self.get_active_palette(state)
        affinity = state['function_affinity']

        top_indices = jnp.argsort(affinity)[-5:][::-1]
        top_affinities = [(int(i), float(affinity[i])) for i in top_indices]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'phase': state['phase'],
            'generation': state['generation'],
            'top_affinity_functions': top_affinities,
            'sin_affinity': float(affinity[4]),
            'avg_affinity': float(jnp.mean(affinity)),
            'stagnation_count': state['stagnation_count'],
            'neuronal_groups': state['neuronal_groups'],
            'antagonistic_pairs': state['antagonistic_pairs'],
            'selection_events': state['selection_events'],
            'total_pruned': len(state['pruned_by_antagonism']),
        }
