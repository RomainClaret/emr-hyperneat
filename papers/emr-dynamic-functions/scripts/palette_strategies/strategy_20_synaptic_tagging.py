"""Strategy 20: Synaptic Tagging and Capture.

Implements synaptic tagging and capture mechanism for palette evolution.

Biological Basis (Frey & Morris, 1997):
- Local synaptic activity creates "tags" at specific synapses
- Tags are transient and decay over time
- Global events (e.g., dopamine release) trigger "capture"
- Only tagged synapses get strengthened by capture
- Requires both local activity AND global success

For palette evolution:
- Local tag: Function correlates with fitness improvement (local event)
- Tag decay: Tags weaken over time if not captured
- Capture: Global fitness breakthrough "captures" tagged functions
- Permanent protection: Captured functions are protected from removal

Key mechanisms:
1. Tag creation: Functions active during local improvement get tagged
2. Tag decay: Tags decay each generation
3. Capture window: Tags can be captured within N generations
4. Capture event: Major fitness improvement captures all active tags
5. Captured functions: Get permanent affinity boost

Expected improvement over Hebbian:
- Fewer false positives (requires BOTH local correlation AND global confirmation)
- More selective protection (only truly important functions get captured)
- Temporal credit assignment (links local events to global outcomes)
- Robust to noise (random correlations don't get captured)
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


class SynapticTaggingStrategy(PaletteEvolutionStrategy):
    """Synaptic tagging and capture for palette evolution.

    Two-stage learning: local tags must be captured by global success.
    """

    name = "synaptic_tagging"
    description = "Two-stage tagging and capture mechanism"

    def __init__(
        self,
        # Tag creation parameters
        tag_threshold: float = 0.15,            # Local improvement to create tag
        tag_strength_per_event: float = 0.3,    # Strength added per tagging event
        max_tag_strength: float = 1.0,          # Maximum tag strength
        # Tag decay parameters
        tag_decay_rate: float = 0.15,           # Tags decay per generation
        tag_min_threshold: float = 0.1,         # Below this, tag is removed
        # Capture parameters
        capture_window: int = 5,                # Gens a tag is valid for capture
        capture_threshold: float = 0.30,        # Global improvement for capture
        capture_efficiency: float = 0.5,        # Fraction of tag converted to affinity
        capture_bonus: float = 0.2,             # Bonus affinity on capture
        # Captured function parameters
        captured_protection: float = 0.8,       # Captured functions are highly protected
        captured_affinity_min: float = 0.7,     # Minimum affinity for captured functions
        # Base learning parameters
        affinity_learning_rate: float = 0.10,   # Base affinity learning rate
        mutation_rate: float = 0.15,            # Base mutation rate
        # Protection
        affinity_protection_threshold: float = 0.55,
        # General
        min_active: int = 2,
        initial_palette: List[int] = None,
    ):
        """Initialize Synaptic Tagging strategy.

        Args:
            tag_threshold: Local fitness improvement needed to create tag
            tag_strength_per_event: How much tag strength per event
            tag_decay_rate: Rate at which tags decay
            capture_window: Generations before tag expires
            capture_threshold: Global improvement needed for capture
            capture_efficiency: Fraction of tag converted to permanent affinity
            capture_bonus: Additional affinity boost on capture
        """
        # Tag creation
        self.tag_threshold = tag_threshold
        self.tag_strength_per_event = tag_strength_per_event
        self.max_tag_strength = max_tag_strength

        # Tag decay
        self.tag_decay_rate = tag_decay_rate
        self.tag_min_threshold = tag_min_threshold

        # Capture
        self.capture_window = capture_window
        self.capture_threshold = capture_threshold
        self.capture_efficiency = capture_efficiency
        self.capture_bonus = capture_bonus

        # Captured function handling
        self.captured_protection = captured_protection
        self.captured_affinity_min = captured_affinity_min

        # Base learning
        self.affinity_learning_rate = affinity_learning_rate
        self.mutation_rate = mutation_rate

        # Protection
        self.affinity_protection_threshold = affinity_protection_threshold

        # General
        self.min_active = min_active
        self.initial_palette = initial_palette or DEFAULT_PALETTE_INDICES

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with tagging and capture systems."""
        initial = config.get('initial_palette', self.initial_palette)
        mask = create_initial_palette_mask(initial)

        # Function affinity (regular learning)
        function_affinity = jnp.ones(NUM_ACTIVATIONS) * 0.5

        # Synaptic tags: [i] = current tag strength for function i
        synaptic_tags = jnp.zeros(NUM_ACTIVATIONS)

        # Tag creation time: [i] = generation when tag was created (-1 = no tag)
        tag_creation_gen = jnp.ones(NUM_ACTIVATIONS) * -100  # -100 = never tagged

        # Captured functions: set of indices that have been captured
        captured_functions = set()

        return {
            'mask': mask,
            'rng_key': jax.random.PRNGKey(seed + 202020),
            'generation': 0,
            'stagnation_count': 0,
            'best_fitness_seen': 0.0,
            'strategy_name': self.name,
            # Tagging system
            'function_affinity': function_affinity,
            'synaptic_tags': synaptic_tags,
            'tag_creation_gen': tag_creation_gen,
            'captured_functions': captured_functions,
            # Tracking
            'fitness_history': [],
            'fitness_ema': 0.5,
            'tag_events': 0,
            'capture_events': 0,
            'functions_captured_this_run': [],
        }

    def get_active_palette(self, state: Dict[str, Any]) -> List[int]:
        """Return current active palette indices."""
        return mask_to_indices(state['mask'])

    def _create_tags(
        self,
        tags: jnp.ndarray,
        tag_gen: jnp.ndarray,
        mask: jnp.ndarray,
        fitness_improvement: float,
        generation: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Create or strengthen tags for active functions.

        Tags are created when local fitness improves above threshold.

        Returns:
            (new_tags, new_tag_gen, n_tagged)
        """
        if fitness_improvement < self.tag_threshold:
            return tags, tag_gen, 0

        new_tags = tags.copy()
        new_tag_gen = tag_gen.copy()
        n_tagged = 0

        active = (mask > 0.5).astype(jnp.float32)

        # Strengthen tags for active functions
        for i in range(NUM_ACTIVATIONS):
            if float(active[i]) > 0.5:
                # Calculate tag strength increase based on fitness improvement
                strength_increase = self.tag_strength_per_event * (
                    fitness_improvement / max(self.tag_threshold, 0.01)
                )
                new_strength = float(tags[i]) + strength_increase

                new_tags = new_tags.at[i].set(
                    min(self.max_tag_strength, new_strength)
                )

                # Update tag creation time (or refresh if already tagged)
                new_tag_gen = new_tag_gen.at[i].set(float(generation))
                n_tagged += 1

        return new_tags, new_tag_gen, n_tagged

    def _decay_tags(
        self,
        tags: jnp.ndarray,
        tag_gen: jnp.ndarray,
        generation: int,
    ) -> jnp.ndarray:
        """Apply tag decay over time.

        Tags decay each generation. Old tags decay faster.
        """
        new_tags = tags.copy()

        for i in range(NUM_ACTIVATIONS):
            if float(tags[i]) < self.tag_min_threshold:
                continue

            # Age of tag in generations
            tag_age = generation - int(tag_gen[i])

            # Decay rate increases with age
            effective_decay = self.tag_decay_rate * (1.0 + 0.1 * tag_age)

            new_strength = float(tags[i]) * (1.0 - effective_decay)

            if new_strength < self.tag_min_threshold:
                new_strength = 0.0

            new_tags = new_tags.at[i].set(new_strength)

        return new_tags

    def _attempt_capture(
        self,
        tags: jnp.ndarray,
        tag_gen: jnp.ndarray,
        affinity: jnp.ndarray,
        captured: Set[int],
        mask: jnp.ndarray,
        fitness_improvement: float,
        generation: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Set[int], List[int]]:
        """Attempt to capture tagged functions on global success.

        Capture occurs when:
        1. Global fitness improvement exceeds threshold
        2. Function has active tag
        3. Tag is within capture window

        Returns:
            (new_tags, new_affinity, new_captured, newly_captured_list)
        """
        if fitness_improvement < self.capture_threshold:
            return tags, affinity, captured, []

        new_tags = tags.copy()
        new_affinity = affinity.copy()
        new_captured = captured.copy()
        newly_captured = []

        active = (mask > 0.5).astype(jnp.float32)

        for i in range(NUM_ACTIVATIONS):
            if i in captured:
                # Already captured, just maintain
                continue

            tag_strength = float(tags[i])
            tag_age = generation - int(tag_gen[i])

            # Check if tag is valid for capture
            if tag_strength >= self.tag_min_threshold and tag_age <= self.capture_window:
                # Capture this function!
                # Convert tag to permanent affinity
                affinity_boost = (
                    tag_strength * self.capture_efficiency +
                    self.capture_bonus
                )
                new_affinity = new_affinity.at[i].set(
                    max(self.captured_affinity_min, float(affinity[i]) + affinity_boost)
                )

                # Mark as captured
                new_captured.add(i)
                newly_captured.append(i)

                # Clear the tag (it's been consumed)
                new_tags = new_tags.at[i].set(0.0)

        return new_tags, new_affinity, new_captured, newly_captured

    def _update_affinity(
        self,
        affinity: jnp.ndarray,
        mask: jnp.ndarray,
        captured: Set[int],
        fitness_signal: float,
    ) -> jnp.ndarray:
        """Update affinity with protection for captured functions."""
        new_affinity = affinity.copy()
        active = (mask > 0.5).astype(jnp.float32)

        for i in range(NUM_ACTIVATIONS):
            if i in captured:
                # Captured functions are protected - only allow positive updates
                if fitness_signal > 0 and float(active[i]) > 0.5:
                    delta = self.affinity_learning_rate * 0.3 * fitness_signal
                    new_affinity = new_affinity.at[i].set(
                        min(0.95, float(new_affinity[i]) + delta)
                    )
                # Ensure minimum affinity
                if float(new_affinity[i]) < self.captured_affinity_min:
                    new_affinity = new_affinity.at[i].set(self.captured_affinity_min)
            else:
                # Normal learning for non-captured functions
                if float(active[i]) > 0.5:
                    if fitness_signal >= 0:
                        delta = self.affinity_learning_rate * fitness_signal
                    else:
                        delta = self.affinity_learning_rate * 0.3 * fitness_signal
                    new_affinity = new_affinity.at[i].set(
                        max(0.05, min(0.95, float(new_affinity[i]) + delta))
                    )

        return new_affinity

    def _compute_effective_protection(
        self,
        affinity: jnp.ndarray,
        tags: jnp.ndarray,
        captured: Set[int],
    ) -> jnp.ndarray:
        """Compute effective protection combining affinity, tags, and capture status."""
        protection = affinity.copy()

        for i in range(NUM_ACTIVATIONS):
            if i in captured:
                # Captured functions have high protection
                protection = protection.at[i].set(self.captured_protection)
            else:
                # Tags provide some protection
                tag_boost = float(tags[i]) * 0.2
                protection = protection.at[i].set(
                    min(0.95, float(protection[i]) + tag_boost)
                )

        return protection

    def _mutate_palette(
        self,
        key: jax.random.PRNGKey,
        mask: jnp.ndarray,
        protection: jnp.ndarray,
        captured: Set[int],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Apply mutation with protection for captured functions."""
        key1, key2 = jax.random.split(key)

        new_mask = mask.copy()
        activated = []
        deactivated = []
        protection_info = {}

        activate_probs = jax.random.uniform(key1, (NUM_ACTIVATIONS,))
        deactivate_probs = jax.random.uniform(key2, (NUM_ACTIVATIONS,))

        for i in range(NUM_ACTIVATIONS):
            prot = float(protection[i])
            is_captured = i in captured

            if mask[i] < 0.5:
                # Inactive: maybe activate
                # Captured functions that were deactivated can be re-activated easily
                if is_captured:
                    effective_rate = self.mutation_rate * 2.0  # High reactivation chance
                else:
                    effective_rate = self.mutation_rate * (0.5 + prot)

                if activate_probs[i] < effective_rate:
                    new_mask = new_mask.at[i].set(1.0)
                    activated.append(i)
            else:
                # Active: maybe deactivate
                if is_captured:
                    # Captured functions are almost never deactivated
                    deact_rate = self.mutation_rate * 0.02
                    protection_info[i] = f"captured (prot={prot:.2f})"
                elif prot >= self.affinity_protection_threshold:
                    # Protected by high affinity/tags
                    deact_rate = self.mutation_rate * 0.1
                    protection_info[i] = f"protected (prot={prot:.2f})"
                else:
                    # Vulnerable
                    deact_rate = self.mutation_rate * (1.0 - prot)

                if deactivate_probs[i] < deact_rate:
                    new_mask = new_mask.at[i].set(0.0)
                    deactivated.append(i)

        # Ensure minimum active
        if jnp.sum(new_mask > 0.5) < self.min_active:
            new_mask = mask
            activated = []
            deactivated = []

        return new_mask, {
            'activated': activated,
            'deactivated': deactivated,
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
        """Update with synaptic tagging and capture mechanism."""
        key, subkey = jax.random.split(state['rng_key'])

        # Track improvement
        improved = best_fitness > state['best_fitness_seen']
        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # Compute fitness signal
        alpha = 0.2
        new_fitness_ema = (1 - alpha) * state['fitness_ema'] + alpha * best_fitness

        # Local fitness improvement (for tagging)
        local_improvement = best_fitness - prev_best_fitness

        # Global fitness improvement (for capture)
        global_improvement = best_fitness - state['fitness_ema']

        # Normalized fitness signal
        fitness_signal = (best_fitness - new_fitness_ema) / max(0.1, new_fitness_ema)
        fitness_signal = max(-1.0, min(1.0, fitness_signal))

        # Step 1: Decay existing tags
        decayed_tags = self._decay_tags(
            state['synaptic_tags'],
            state['tag_creation_gen'],
            generation,
        )

        # Step 2: Create/strengthen tags on local improvement
        new_tags, new_tag_gen, n_tagged = self._create_tags(
            decayed_tags,
            state['tag_creation_gen'],
            state['mask'],
            local_improvement,
            generation,
        )

        # Step 3: Attempt capture on global success
        new_tags, new_affinity, new_captured, newly_captured = self._attempt_capture(
            new_tags,
            new_tag_gen,
            state['function_affinity'],
            state['captured_functions'],
            state['mask'],
            global_improvement,
            generation,
        )

        # Step 4: Update affinity normally
        new_affinity = self._update_affinity(
            new_affinity,
            state['mask'],
            new_captured,
            fitness_signal,
        )

        # Compute protection scores
        protection = self._compute_effective_protection(
            new_affinity, new_tags, new_captured
        )

        # Apply mutation
        new_mask, mutation_info = self._mutate_palette(
            subkey, state['mask'], protection, new_captured
        )

        palette_changed = not jnp.allclose(state['mask'], new_mask)

        # Update fitness history
        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        # Update counters
        tag_events = state['tag_events'] + n_tagged
        capture_events = state['capture_events'] + len(newly_captured)
        functions_captured = state['functions_captured_this_run'] + newly_captured

        new_state = {
            'mask': new_mask,
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            # Tagging system
            'function_affinity': new_affinity,
            'synaptic_tags': new_tags,
            'tag_creation_gen': new_tag_gen,
            'captured_functions': new_captured,
            # Tracking
            'fitness_history': fitness_history,
            'fitness_ema': new_fitness_ema,
            'tag_events': tag_events,
            'capture_events': capture_events,
            'functions_captured_this_run': functions_captured,
        }

        active_palette = mask_to_indices(new_mask)
        tagged_funcs = [i for i in range(NUM_ACTIVATIONS) if float(new_tags[i]) >= self.tag_min_threshold]

        metrics = {
            'palette_changed': palette_changed,
            'current_palette': active_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # Tagging metrics
            'n_tagged_this_gen': n_tagged,
            'n_captured_this_gen': len(newly_captured),
            'newly_captured': newly_captured,
            'total_captured': list(new_captured),
            'n_total_captured': len(new_captured),
            'tagged_functions': tagged_funcs,
            'n_tagged_functions': len(tagged_funcs),
            # Affinity stats
            'avg_affinity': float(jnp.mean(new_affinity)),
            'max_affinity': float(jnp.max(new_affinity)),
            'sin_affinity': float(new_affinity[4]),
            'sin_tag': float(new_tags[4]),
            'sin_captured': 4 in new_captured,
            # Global stats
            'total_tag_events': tag_events,
            'total_capture_events': capture_events,
            'local_improvement': local_improvement,
            'global_improvement': global_improvement,
        }
        metrics.update(mutation_info)

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with tagging and capture stats."""
        palette = self.get_active_palette(state)
        affinity = state['function_affinity']
        tags = state['synaptic_tags']
        captured = state['captured_functions']

        # Top captured functions
        captured_list = list(captured)
        captured_affinities = [(int(i), float(affinity[i])) for i in captured_list]

        # Currently tagged
        tagged = [i for i in range(NUM_ACTIVATIONS) if float(tags[i]) >= self.tag_min_threshold]

        return {
            'strategy': self.name,
            'active_palette': palette,
            'palette_size': len(palette),
            'has_sin': 4 in palette,
            'sin_captured': 4 in captured,
            'generation': state['generation'],
            'captured_functions': captured_affinities,
            'n_captured': len(captured),
            'tagged_functions': tagged,
            'n_tagged': len(tagged),
            'sin_affinity': float(affinity[4]),
            'sin_tag': float(tags[4]),
            'avg_affinity': float(jnp.mean(affinity)),
            'stagnation_count': state['stagnation_count'],
            'total_tag_events': state['tag_events'],
            'total_capture_events': state['capture_events'],
        }
