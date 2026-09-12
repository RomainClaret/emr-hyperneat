"""Strategy 109: GRN Cross-Domain v2 Dual.

Combines cross_domain_v2_dual (#103) with genetic_regulatory_network (#33)
mechanics. GRN attractors lock sin-extreme pairings into stable expression
patterns through evolved regulatory circuits.

Problem with pure multiplicative coupling:
- Cross-affinity can still decay over extended non-use
- No structural locking mechanism
- Domain shifts can erode even strong affinities

Key Innovation:
- GRN regulation creates explicit activation-aggregation links
- Sin-extreme pairings form stable regulatory attractors
- Expression-based selection reinforces proven pairings
- Regulatory circuits "remember" successful combinations

Bio inspiration: Gene regulatory networks create stable expression states
through evolved activation/inhibition circuits. Similarly, sin-extreme
pairings can be locked into regulatory attractors that resist perturbation.

Expected: Sin-extreme pairings survive domain shifts through regulatory locking.
"""

from typing import Dict, Any, List, Optional, Tuple
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


class GRNCrossV2DualStrategy(PaletteEvolutionStrategy):
    """GRN-enhanced cross-domain multiplicative dual palette evolution.

    Combines multiplicative cross-domain affinity
    with GRN regulatory circuits. Sin-extreme pairings form stable
    regulatory attractors.

    Critical innovation: Regulatory circuits create structural locking
    beyond affinity - pairings are "wired in" through evolved regulation.
    """

    name = "grn_cross_v2_dual"
    description = "Dual: Multiplicative cross-domain with GRN attractor locking"

    def __init__(
        self,
        # === GRN PARAMETERS (NEW from genetic_regulatory_network) ===
        hill_coefficient: float = 2.0,             # Sigmoidal steepness
        regulation_learning_rate: float = 0.08,    # How fast links adapt
        regulation_decay: float = 0.98,            # Passive link decay
        regulation_max: float = 1.5,               # Max regulation strength
        sin_extreme_activation_link: float = 0.5,  # Initial sin-extreme link
        multiplicative_regulation_update: bool = True,  # Multiplicative GRN updates
        expression_influence: float = 0.3,         # How much expression affects selection
        # === MULTIPLICATIVE CROSS-DOMAIN (from #103) ===
        multiplicative_update: bool = True,
        cross_affinity_growth_factor: float = 1.15,
        cross_affinity_decay_factor: float = 0.98,
        sin_extreme_multiplier: float = 1.5,
        mutual_capture_bonus: float = 0.4,
        # === Affinity parameters ===
        act_affinity_lr: float = 0.12,
        agg_affinity_lr: float = 0.10,
        affinity_decay: float = 0.98,
        base_cross_learning_rate: float = 0.10,
        # === Tagging parameters ===
        tag_threshold: float = 0.5,
        agg_tag_threshold: float = 0.45,
        tag_decay: float = 0.9,
        capture_window: int = 5,
        captured_protection: float = 0.8,
        extreme_tag_boost: float = 1.3,
        # === Constraints ===
        min_active_act: int = 2,
        min_active_agg: int = 1,
        max_active_act: int = 8,
        max_active_agg: int = 4,
        min_diversity_act: int = 4,
        min_diversity_agg: int = 2,
        # === Initial palettes ===
        initial_act_palette: List[int] = None,
        initial_agg_palette: List[int] = None,
    ):
        """Initialize GRN Cross-Domain v2 strategy."""
        # GRN parameters (NEW)
        self.hill_coefficient = hill_coefficient
        self.regulation_learning_rate = regulation_learning_rate
        self.regulation_decay = regulation_decay
        self.regulation_max = regulation_max
        self.sin_extreme_activation_link = sin_extreme_activation_link
        self.multiplicative_regulation_update = multiplicative_regulation_update
        self.expression_influence = expression_influence

        # Multiplicative cross-domain
        self.multiplicative_update = multiplicative_update
        self.cross_affinity_growth_factor = cross_affinity_growth_factor
        self.cross_affinity_decay_factor = cross_affinity_decay_factor
        self.sin_extreme_multiplier = sin_extreme_multiplier
        self.mutual_capture_bonus = mutual_capture_bonus

        # Affinity
        self.act_affinity_lr = act_affinity_lr
        self.agg_affinity_lr = agg_affinity_lr
        self.affinity_decay = affinity_decay
        self.base_cross_learning_rate = base_cross_learning_rate

        # Tagging
        self.tag_threshold = tag_threshold
        self.agg_tag_threshold = agg_tag_threshold
        self.tag_decay = tag_decay
        self.capture_window = capture_window
        self.captured_protection = captured_protection
        self.extreme_tag_boost = extreme_tag_boost

        # Constraints
        self.min_active_act = min_active_act
        self.min_active_agg = min_active_agg
        self.max_active_act = max_active_act
        self.max_active_agg = max_active_agg
        self.min_diversity_act = min_diversity_act
        self.min_diversity_agg = min_diversity_agg

        # Initial
        self.initial_act_palette = initial_act_palette or DEFAULT_PALETTE_INDICES
        self.initial_agg_palette = initial_agg_palette or DEFAULT_AGG_PALETTE_INDICES

    def _initialize_cross_regulation(self) -> jnp.ndarray:
        """Initialize cross-domain regulatory network.

        This creates activation links between activations and aggregations.
        Sin (4) starts with activation links to extreme aggs (2, 3).
        """
        # Cross-regulation: activations regulate aggregations and vice versa
        cross_reg = jnp.zeros((NUM_ACTIVATIONS, NUM_AGGREGATIONS))

        # Sin-extreme initial links
        for j in CORE_EXTREME_AGGS:
            cross_reg = cross_reg.at[4, j].set(self.sin_extreme_activation_link)

        return cross_reg

    def initialize(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Initialize state with GRN regulatory tracking."""
        initial_act = config.get('initial_palette', self.initial_act_palette)
        initial_agg = config.get('initial_agg_palette', self.initial_agg_palette)

        act_mask = create_initial_palette_mask(initial_act)
        agg_mask = create_initial_agg_palette_mask(initial_agg)

        # Affinities
        act_affinities = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_affinities = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_affinities = act_affinities.at[i].set(0.5)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_affinities = agg_affinities.at[i].set(0.5)

        # Tagging state
        act_tags = jnp.zeros(NUM_ACTIVATIONS)
        agg_tags = jnp.zeros(NUM_AGGREGATIONS)
        act_captured = jnp.zeros(NUM_ACTIVATIONS)
        agg_captured = jnp.zeros(NUM_AGGREGATIONS)

        # Cross-domain affinity (starts higher for sin-extreme)
        cross_affinity = jnp.ones((NUM_ACTIVATIONS, NUM_AGGREGATIONS)) * 0.5
        for j in CORE_EXTREME_AGGS:
            cross_affinity = cross_affinity.at[4, j].set(0.6)

        # GRN state (NEW)
        cross_regulation = self._initialize_cross_regulation()
        act_expression = jnp.ones(NUM_ACTIVATIONS) * 0.3
        agg_expression = jnp.ones(NUM_AGGREGATIONS) * 0.3
        for i in initial_act:
            if 0 <= i < NUM_ACTIVATIONS:
                act_expression = act_expression.at[i].set(0.6)
        for i in initial_agg:
            if 0 <= i < NUM_AGGREGATIONS:
                agg_expression = agg_expression.at[i].set(0.6)

        return {
            # Masks
            'act_mask': act_mask,
            'agg_mask': agg_mask,
            # Affinities
            'act_affinities': act_affinities,
            'agg_affinities': agg_affinities,
            # Tagging
            'act_tags': act_tags,
            'agg_tags': agg_tags,
            'act_captured': act_captured,
            'agg_captured': agg_captured,
            'tag_history': [],
            # Cross-domain
            'cross_affinity': cross_affinity,
            # GRN state (NEW)
            'cross_regulation': cross_regulation,
            'act_expression': act_expression,
            'agg_expression': agg_expression,
            # Stats
            'capture_events': 0,
            'mutual_capture_events': 0,
            'multiplicative_growth_events': 0,
            'regulation_strengthening_events': 0,
            'attractor_lock_events': 0,
            'diversity_rescues': 0,
            # General
            'rng_key': jax.random.PRNGKey(seed + 1090000),
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

    def _hill_function(self, x: float, positive: bool = True) -> float:
        """Hill function for sigmoidal response."""
        n = self.hill_coefficient
        K = 0.5  # Half-max constant
        if positive:
            return (x ** n) / (K ** n + x ** n + 1e-8)
        else:
            return (K ** n) / (K ** n + x ** n + 1e-8)

    def _update_expression(
        self,
        act_expression: jnp.ndarray,
        agg_expression: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_regulation: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update expression levels based on regulatory network.

        Activations and aggregations regulate each other through cross_regulation.
        """
        new_act_expr = act_expression * 0.9  # Decay
        new_agg_expr = agg_expression * 0.9

        # Activation expression influenced by aggregation regulation
        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                # Sum regulation from active aggregations
                reg_sum = 0.0
                for j in range(NUM_AGGREGATIONS):
                    if agg_mask[j] > 0.5:
                        # Regulation is bidirectional - aggs can boost acts
                        reg_sum += float(cross_regulation[i, j]) * float(agg_expression[j])

                # Apply Hill function
                boost = self._hill_function(reg_sum)
                new_act_expr = new_act_expr.at[i].set(
                    min(1.0, float(new_act_expr[i]) + boost * 0.2)
                )

        # Aggregation expression influenced by activation regulation
        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                reg_sum = 0.0
                for i in range(NUM_ACTIVATIONS):
                    if act_mask[i] > 0.5:
                        reg_sum += float(cross_regulation[i, j]) * float(act_expression[i])

                boost = self._hill_function(reg_sum)
                new_agg_expr = new_agg_expr.at[j].set(
                    min(1.0, float(new_agg_expr[j]) + boost * 0.2)
                )

        return new_act_expr, new_agg_expr

    def _update_cross_regulation(
        self,
        cross_regulation: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, int, int]:
        """Update cross-regulatory links based on co-activity and fitness.

        KEY INNOVATION: Successful sin-extreme co-activation strengthens
        regulatory links, creating stable attractor states.
        """
        new_reg = cross_regulation * self.regulation_decay
        strengthening_events = 0
        attractor_locks = 0

        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        current = float(new_reg[i, j])

                        if self.multiplicative_regulation_update:
                            # Multiplicative update for stronger locking
                            growth = 1.0 + self.regulation_learning_rate * fitness_delta

                            # Sin-extreme gets extra boost
                            if i == 4 and j in CORE_EXTREME_AGGS:
                                growth *= self.sin_extreme_multiplier
                                # Check for attractor lock
                                if current > 0.8:
                                    attractor_locks += 1

                            new_value = current * growth
                        else:
                            # Additive update
                            delta = self.regulation_learning_rate * fitness_delta
                            if i == 4 and j in CORE_EXTREME_AGGS:
                                delta *= self.sin_extreme_multiplier
                            new_value = current + delta

                        new_reg = new_reg.at[i, j].set(
                            min(self.regulation_max, new_value)
                        )
                        strengthening_events += 1
        else:
            # Slight decay but preserve structure
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        current = float(new_reg[i, j])
                        # Protected decay for sin-extreme (attractor stability)
                        if i == 4 and j in CORE_EXTREME_AGGS:
                            new_value = current * 0.995  # Very slow decay
                        else:
                            new_value = current * 0.98

                        new_reg = new_reg.at[i, j].set(max(0.0, new_value))

        return new_reg, strengthening_events, attractor_locks

    def _update_tags(
        self,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        cross_regulation: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update tags with regulation influence."""
        new_act_tags = act_tags * self.tag_decay
        new_agg_tags = agg_tags * self.tag_decay

        for i in range(NUM_ACTIVATIONS):
            if act_mask[i] > 0.5:
                tag_strength = 1.0
                if i == 4:
                    tag_strength *= self.extreme_tag_boost

                # Cross-affinity boost
                max_cross = float(jnp.max(cross_affinity[i, :]))
                tag_strength *= (1 + 0.2 * max_cross)

                # REGULATION BOOST (NEW) - strong regulation = faster tagging
                max_reg = float(jnp.max(cross_regulation[i, :]))
                tag_strength *= (1 + 0.3 * max_reg)

                new_act_tags = new_act_tags.at[i].set(
                    min(1.0, new_act_tags[i] + tag_strength * 0.3)
                )

        for j in range(NUM_AGGREGATIONS):
            if agg_mask[j] > 0.5:
                tag_strength = 1.0
                if j in CORE_EXTREME_AGGS:
                    tag_strength *= self.extreme_tag_boost

                max_cross = float(jnp.max(cross_affinity[:, j]))
                tag_strength *= (1 + 0.2 * max_cross)

                max_reg = float(jnp.max(cross_regulation[:, j]))
                tag_strength *= (1 + 0.3 * max_reg)

                new_agg_tags = new_agg_tags.at[j].set(
                    min(1.0, new_agg_tags[j] + tag_strength * 0.3)
                )

        return new_act_tags, new_agg_tags

    def _attempt_capture_with_mutual_bonus(
        self,
        act_tags: jnp.ndarray,
        agg_tags: jnp.ndarray,
        act_captured: jnp.ndarray,
        agg_captured: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        cross_regulation: jnp.ndarray,
        tag_history: List,
        generation: int,
        improved: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int]:
        """Attempt capture with mutual bonus and regulation boost."""
        new_act_captured = act_captured.copy()
        new_agg_captured = agg_captured.copy()
        new_cross = cross_affinity.copy()
        new_reg = cross_regulation.copy()
        capture_count = 0
        mutual_capture_count = 0

        if not improved:
            return new_act_captured, new_agg_captured, new_cross, new_reg, 0, 0

        act_captured_this_round = []
        agg_captured_this_round = []

        for hist_gen, hist_act_tags, hist_agg_tags in tag_history:
            if generation - hist_gen <= self.capture_window:
                for i in range(NUM_ACTIVATIONS):
                    if hist_act_tags[i] > self.tag_threshold and new_act_captured[i] < 0.5:
                        new_act_captured = new_act_captured.at[i].set(1.0)
                        capture_count += 1
                        act_captured_this_round.append(i)

                for j in range(NUM_AGGREGATIONS):
                    if hist_agg_tags[j] > self.agg_tag_threshold and new_agg_captured[j] < 0.5:
                        new_agg_captured = new_agg_captured.at[j].set(1.0)
                        capture_count += 1
                        agg_captured_this_round.append(j)

        # MUTUAL CAPTURE BONUS with regulation lock
        for i in act_captured_this_round:
            for j in agg_captured_this_round:
                current_aff = float(new_cross[i, j])
                current_reg = float(new_reg[i, j])

                bonus_factor = 1.0 + self.mutual_capture_bonus
                if i == 4 and j in CORE_EXTREME_AGGS:
                    bonus_factor *= self.sin_extreme_multiplier

                new_cross = new_cross.at[i, j].set(min(1.0, current_aff * bonus_factor))
                # Also boost regulation (attractor locking)
                new_reg = new_reg.at[i, j].set(
                    min(self.regulation_max, current_reg * bonus_factor)
                )
                mutual_capture_count += 1

        return new_act_captured, new_agg_captured, new_cross, new_reg, capture_count, mutual_capture_count

    def _update_affinities_multiplicative(
        self,
        act_affinities: jnp.ndarray,
        agg_affinities: jnp.ndarray,
        act_mask: jnp.ndarray,
        agg_mask: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        cross_regulation: jnp.ndarray,
        fitness_delta: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """Update affinities with multiplicative cross-domain dynamics."""
        growth_events = 0

        new_act_aff = act_affinities * self.affinity_decay
        new_agg_aff = agg_affinities * self.affinity_decay

        if fitness_delta > 0:
            for i in range(NUM_ACTIVATIONS):
                if act_mask[i] > 0.5:
                    new_act_aff = new_act_aff.at[i].set(
                        min(1.0, new_act_aff[i] + self.act_affinity_lr * fitness_delta)
                    )
            for j in range(NUM_AGGREGATIONS):
                if agg_mask[j] > 0.5:
                    new_agg_aff = new_agg_aff.at[j].set(
                        min(1.0, new_agg_aff[j] + self.agg_affinity_lr * fitness_delta)
                    )

        # MULTIPLICATIVE CROSS-DOMAIN UPDATE
        new_cross = cross_affinity.copy()

        if self.multiplicative_update:
            for i in range(NUM_ACTIVATIONS):
                for j in range(NUM_AGGREGATIONS):
                    current = float(new_cross[i, j])
                    reg_strength = float(cross_regulation[i, j])

                    if act_mask[i] > 0.5 and agg_mask[j] > 0.5:
                        if fitness_delta > 0:
                            growth_factor = self.cross_affinity_growth_factor

                            # Regulation boost - stronger regulation = stronger growth
                            growth_factor *= (1 + reg_strength * 0.2)

                            if i == 4 and j in CORE_EXTREME_AGGS:
                                growth_factor *= self.sin_extreme_multiplier

                            new_value = current * growth_factor
                            new_cross = new_cross.at[i, j].set(min(1.0, new_value))
                            growth_events += 1
                        else:
                            # PROTECTED DECAY for regulated pairs
                            decay = self.cross_affinity_decay_factor
                            if reg_strength > 0.5:
                                decay = decay ** 0.5  # Slower decay
                            new_value = max(0.2, current * decay)
                            new_cross = new_cross.at[i, j].set(new_value)
                    else:
                        new_value = max(0.1, current * 0.995)
                        new_cross = new_cross.at[i, j].set(new_value)

        return new_act_aff, new_agg_aff, new_cross, growth_events

    def _select_palette(
        self,
        affinities: jnp.ndarray,
        captured: jnp.ndarray,
        tags: jnp.ndarray,
        expression: jnp.ndarray,
        cross_affinity: jnp.ndarray,
        cross_regulation: jnp.ndarray,
        other_mask: jnp.ndarray,
        n_funcs: int,
        is_act: bool,
        min_active: int,
        max_active: int,
        min_diversity: int,
        key: jax.random.PRNGKey,
        prefer_indices: List[int] = None,
    ) -> Tuple[jnp.ndarray, int]:
        """Select palette with expression and regulation influence."""
        # Base score
        score = affinities + captured * 0.3 + tags * 0.2

        # EXPRESSION INFLUENCE (NEW)
        score = score + expression * self.expression_influence

        # Cross-domain influence
        for i in range(n_funcs):
            if is_act:
                cross_influence = 0.0
                reg_influence = 0.0
                for j in range(NUM_AGGREGATIONS):
                    if other_mask[j] > 0.5:
                        cross_influence = max(cross_influence, float(cross_affinity[i, j]))
                        reg_influence = max(reg_influence, float(cross_regulation[i, j]))
            else:
                cross_influence = 0.0
                reg_influence = 0.0
                for i_act in range(NUM_ACTIVATIONS):
                    if other_mask[i_act] > 0.5:
                        cross_influence = max(cross_influence, float(cross_affinity[i_act, i]))
                        reg_influence = max(reg_influence, float(cross_regulation[i_act, i]))

            score = score.at[i].set(score[i] + cross_influence * 0.25 + reg_influence * 0.3)

        if prefer_indices:
            for i in prefer_indices:
                if 0 <= i < n_funcs:
                    score = score.at[i].set(score[i] + 0.5)

        target_size = min(max(min_diversity, min_active), max_active, n_funcs)
        top_indices = jnp.argsort(score)[-target_size:]

        mask = jnp.zeros(n_funcs)
        for idx in top_indices:
            mask = mask.at[int(idx)].set(1.0)

        # Diversity rescue
        active_count = int(jnp.sum(mask > 0.5))
        diversity_rescue = 0
        if active_count < min_diversity:
            k1, _ = jax.random.split(key)
            inactive = [i for i in range(n_funcs) if mask[i] < 0.5]
            needed = min_diversity - active_count
            if inactive and needed > 0:
                to_add = jax.random.choice(k1, jnp.array(inactive), shape=(min(needed, len(inactive)),), replace=False)
                for i in to_add:
                    mask = mask.at[int(i)].set(1.0)
                    diversity_rescue += 1

        return mask, diversity_rescue

    def _compute_extreme_ratio(self, agg_mask: jnp.ndarray) -> float:
        """Compute extreme/averaging ratio."""
        active_extreme = sum(1 for i in EXTREME_AGGS if agg_mask[i] > 0.5)
        active_averaging = sum(1 for i in AVERAGING_AGGS if agg_mask[i] > 0.5)
        total = active_extreme + active_averaging
        if total == 0:
            return 0.5
        return active_extreme / total

    def post_generation_update(
        self,
        state: Dict[str, Any],
        generation: int,
        best_fitness: float,
        prev_best_fitness: float,
        population_data: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update with GRN-enhanced multiplicative cross-domain dynamics."""
        key, k1, k2 = jax.random.split(state['rng_key'], 3)

        improved = best_fitness > state['best_fitness_seen']
        fitness_delta = best_fitness - prev_best_fitness

        if improved:
            new_stagnation = 0
            new_best = best_fitness
        else:
            new_stagnation = state['stagnation_count'] + 1
            new_best = state['best_fitness_seen']

        # === EXPRESSION UPDATE (NEW) ===
        new_act_expr, new_agg_expr = self._update_expression(
            state['act_expression'], state['agg_expression'],
            state['act_mask'], state['agg_mask'],
            state['cross_regulation']
        )

        # === REGULATION UPDATE (NEW) ===
        new_cross_reg, strengthening_events, attractor_locks = self._update_cross_regulation(
            state['cross_regulation'],
            state['act_mask'], state['agg_mask'],
            state['cross_affinity'], fitness_delta
        )

        # === TAGGING WITH REGULATION ===
        new_act_tags, new_agg_tags = self._update_tags(
            state['act_mask'], state['agg_mask'],
            state['act_tags'], state['agg_tags'],
            state['cross_affinity'], new_cross_reg
        )

        new_tag_history = state['tag_history'] + [(generation, state['act_tags'], state['agg_tags'])]
        if len(new_tag_history) > self.capture_window + 2:
            new_tag_history = new_tag_history[-(self.capture_window + 2):]

        # === CAPTURE WITH MUTUAL BONUS AND REGULATION ===
        (new_act_captured, new_agg_captured, post_capture_cross, post_capture_reg,
         capture_count, mutual_capture_count) = self._attempt_capture_with_mutual_bonus(
            new_act_tags, new_agg_tags,
            state['act_captured'], state['agg_captured'],
            state['cross_affinity'], new_cross_reg,
            new_tag_history, generation, improved
        )

        # === MULTIPLICATIVE AFFINITY UPDATE ===
        new_act_aff, new_agg_aff, new_cross_affinity, growth_events = \
            self._update_affinities_multiplicative(
                state['act_affinities'], state['agg_affinities'],
                state['act_mask'], state['agg_mask'],
                post_capture_cross, post_capture_reg, fitness_delta
            )

        # === PALETTE SELECTION WITH EXPRESSION ===
        new_act_mask, act_diversity_rescue = self._select_palette(
            new_act_aff, new_act_captured, new_act_tags, new_act_expr,
            new_cross_affinity, post_capture_reg, state['agg_mask'],
            NUM_ACTIVATIONS, True,
            self.min_active_act, self.max_active_act, self.min_diversity_act,
            k1, prefer_indices=[4]
        )
        new_agg_mask, agg_diversity_rescue = self._select_palette(
            new_agg_aff, new_agg_captured, new_agg_tags, new_agg_expr,
            new_cross_affinity, post_capture_reg, new_act_mask,
            NUM_AGGREGATIONS, False,
            self.min_active_agg, self.max_active_agg, self.min_diversity_agg,
            k2, prefer_indices=list(CORE_EXTREME_AGGS)
        )

        extreme_ratio = self._compute_extreme_ratio(new_agg_mask)

        act_changed = not jnp.allclose(state['act_mask'], new_act_mask)
        agg_changed = not jnp.allclose(state['agg_mask'], new_agg_mask)

        fitness_history = state['fitness_history'] + [best_fitness]
        if len(fitness_history) > 20:
            fitness_history = fitness_history[-20:]

        new_state = {
            'act_mask': new_act_mask,
            'agg_mask': new_agg_mask,
            'act_affinities': new_act_aff,
            'agg_affinities': new_agg_aff,
            'act_tags': new_act_tags,
            'agg_tags': new_agg_tags,
            'act_captured': new_act_captured,
            'agg_captured': new_agg_captured,
            'tag_history': new_tag_history,
            'cross_affinity': new_cross_affinity,
            # GRN state (NEW)
            'cross_regulation': post_capture_reg,
            'act_expression': new_act_expr,
            'agg_expression': new_agg_expr,
            # Stats
            'capture_events': state['capture_events'] + capture_count,
            'mutual_capture_events': state['mutual_capture_events'] + mutual_capture_count,
            'multiplicative_growth_events': state['multiplicative_growth_events'] + growth_events,
            'regulation_strengthening_events': state['regulation_strengthening_events'] + strengthening_events,
            'attractor_lock_events': state['attractor_lock_events'] + attractor_locks,
            'diversity_rescues': state['diversity_rescues'] + act_diversity_rescue + agg_diversity_rescue,
            # General
            'rng_key': key,
            'generation': generation + 1,
            'stagnation_count': new_stagnation,
            'best_fitness_seen': new_best,
            'strategy_name': self.name,
            'fitness_history': fitness_history,
        }

        act_palette = mask_to_indices(new_act_mask)
        agg_palette = mask_to_indices(new_agg_mask)

        # Sin-extreme regulation strength
        sin_max_reg = float(post_capture_reg[4, 2])
        sin_min_reg = float(post_capture_reg[4, 3])

        metrics = {
            'palette_changed': act_changed,
            'agg_palette_changed': agg_changed,
            'current_palette': act_palette,
            'current_agg_palette': agg_palette,
            'stagnation_count': new_stagnation,
            'fitness_improved': improved,
            # GRN metrics (NEW)
            'sin_max_regulation': sin_max_reg,
            'sin_min_regulation': sin_min_reg,
            'max_regulation': float(jnp.max(post_capture_reg)),
            'mean_regulation': float(jnp.mean(post_capture_reg)),
            'regulation_strengthening_events': new_state['regulation_strengthening_events'],
            'attractor_lock_events': new_state['attractor_lock_events'],
            'sin_expression': float(new_act_expr[4]),
            'max_expression': float(new_agg_expr[2]),
            'min_expression': float(new_agg_expr[3]),
            # Cross-domain metrics
            'sin_max_cross': float(new_cross_affinity[4, 2]),
            'sin_min_cross': float(new_cross_affinity[4, 3]),
            'max_cross_affinity': float(jnp.max(new_cross_affinity)),
            'mean_cross_affinity': float(jnp.mean(new_cross_affinity)),
            'multiplicative_growth_events': new_state['multiplicative_growth_events'],
            'mutual_capture_events': new_state['mutual_capture_events'],
            # Affinity metrics
            'act_mean_affinity': float(jnp.mean(new_act_aff)),
            'agg_mean_affinity': float(jnp.mean(new_agg_aff)),
            'sin_affinity': float(new_act_aff[4]),
            # Tagging metrics
            'sin_tag': float(new_act_tags[4]),
            'sin_captured': bool(new_act_captured[4] > 0.5),
            'capture_events': new_state['capture_events'],
            # Homeostatic
            'extreme_ratio': extreme_ratio,
            # Status
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
        }

        return new_state, metrics

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state summary with GRN status."""
        act_palette = self.get_active_palette(state)
        agg_palette = self.get_active_agg_palette(state)

        return {
            'strategy': self.name,
            'active_palette': act_palette,
            'active_agg_palette': agg_palette,
            'has_sin': 4 in act_palette,
            'has_max': 2 in agg_palette,
            'has_min': 3 in agg_palette,
            # GRN (NEW)
            'sin_max_regulation': float(state['cross_regulation'][4, 2]),
            'sin_min_regulation': float(state['cross_regulation'][4, 3]),
            'max_regulation': float(jnp.max(state['cross_regulation'])),
            'regulation_strengthening_events': state['regulation_strengthening_events'],
            'attractor_lock_events': state['attractor_lock_events'],
            # Cross-domain
            'sin_max_cross': float(state['cross_affinity'][4, 2]),
            'sin_min_cross': float(state['cross_affinity'][4, 3]),
            'max_cross_affinity': float(jnp.max(state['cross_affinity'])),
            'multiplicative_growth_events': state['multiplicative_growth_events'],
            'mutual_capture_events': state['mutual_capture_events'],
            # Affinity
            'sin_captured': bool(state['act_captured'][4] > 0.5),
            'capture_events': state['capture_events'],
            # General
            'generation': state['generation'],
            'stagnation_count': state['stagnation_count'],
        }
