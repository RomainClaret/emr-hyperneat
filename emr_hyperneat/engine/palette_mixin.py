"""Palettes at run time: the checks of a function palette and of a live palette strategy, the strategy's
updates between generations, and palette evolution."""
import inspect
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.engine.functions import (
    ACTIVATION_LIST,
    AGGREGATION_LIST,
    LIVE_STRATEGY_KEY,
    mutate_palette_mask,
    palette_snap_table,
    parse_live_strategy,
)


class PaletteMixin:
    """EMRHyperNEAT's palette methods."""

    def _check_function_palettes(self) -> None:
        """Refuse a palette that names a function outside its list, and an aggregation palette under
        a mode that decodes no per-node aggregation. Runners set df_palette and agg_palette between
        generations, so this runs every generation. 0.1.0 failed with an IndexError on an entry
        beyond the list, wrapped a negative one, and read no aggregation palette.

        Raises:
            ValueError: An entry outside its list, or an aggregation palette nothing would apply.
        """
        for name, palette, names in (('df_palette', self.df_palette, ACTIVATION_LIST),
                                     ('agg_palette', self.agg_palette, AGGREGATION_LIST)):
            if palette is None:
                continue
            values = np.asarray(palette)
            if (
                values.ndim != 1
                or values.size == 0
                or values.min() < 0
                or values.max() >= len(names)
            ):
                raise ValueError(f"{name} must hold indices 0 to {len(names) - 1} of {names}, got "
                                 f"{values.tolist()}")
        no_decode = self.agg_mode in ('disabled', 'global') or (
            self.agg_mode == 'modular' and self.df_modular_config['aggregation_method'] == 'none')
        if self.agg_palette is not None and no_decode and self.hh_aggregation_mode != 'dynamic':
            raise ValueError(
                f"an aggregation palette is set but aggregation.mode {self.agg_mode!r} decodes no "
                "per-node aggregation, so it would be ignored"
            )

    def _check_live_strategy(self) -> None:
        """Refuse the live palette strategy where nothing would apply it.

        Raises:
            ValueError: No per-node activation decode (mode 'cppn_output'), or palette evolution,
                which sets the palette itself.
        """
        if self.dynamic_functions_mode != 'cppn_output':
            raise ValueError(
                f"the live palette strategy sets the palette of the per-node CPPN decode: it needs "
                "dynamic_functions.mode 'cppn_output', got "
                f"{self.dynamic_functions_mode!r}"
            )
        if getattr(self, 'df_palette_evolution', None) is not None:
            raise ValueError(
                "the live palette strategy and dynamic_functions.palette_evolution both set the "
                "palette; configure one"
            )

    def set_palette_strategy(self, strategy: Any) -> None:
        """HMR's live palette strategy: a runner's strategy object sets the palette after each
        generation.

        The object has ``get_active_palette()`` (ACTIVATION_LIST indices), and may have
        ``post_generation_update(generation=..., best_fitness=..., fitness_history=...)``, called
        after each generation's tell, and ``initialize()``, called here when it needs no argument.
        Its palette now becomes the run's. When a later palette is longer, a share of the population
        (``perturbation_rate``) gets ``perturbation_strength`` added to the bias of the CPPN's
        activation output node, which moves the activation channel toward the higher bins. Without a
        ``dynamic_functions.live_strategy`` block the call enables the strategy with HMR's defaults
        (rate 0.15, strength 0.3, no decay), as HMR's did.

        Args:
            strategy: The strategy object.

        Raises:
            ValueError: No per-node activation decode, palette evolution, or a palette outside
                ACTIVATION_LIST.
            TypeError: The object has no ``get_active_palette``.
        """
        self._check_live_strategy()
        if not callable(getattr(strategy, 'get_active_palette', None)):
            raise TypeError("a palette strategy needs get_active_palette() (HMR skipped an object "
                            "without it)")
        if self._live_strategy is None:
            self._live_strategy = parse_live_strategy({'enabled': True})
        self._palette_strategy = strategy
        self._live_generation = 0
        self._live_fitness_history = []
        initialize = getattr(strategy, 'initialize', None)
        if callable(initialize):
            parameters = inspect.signature(initialize).parameters.values()
            if all(
                p.default is not inspect.Parameter.empty
                or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for p in parameters
            ):
                initialize()
        self.df_palette = jnp.asarray(
            [int(i) for i in strategy.get_active_palette()], dtype=jnp.int32
        )
        self._check_function_palettes()

    def update_palette_from_strategy(self, state: Any, best_fitness: float) -> Any:
        """HMR's name: one step of the live palette strategy, after a generation's tell.

        Tells the strategy the generation (counted from 1) and its best fitness, reads its palette,
        and when the palette grew perturbs the population (:meth:`_perturb_activation_outputs`).

        Args:
            state: The state the tell returned.
            best_fitness: The generation's best fitness.

        Returns:
            The state, with the perturbed population when the palette grew.

        Raises:
            ValueError: live_strategy is enabled and no strategy is set, or a palette outside
                ACTIVATION_LIST.
        """
        if self._palette_strategy is None:
            raise ValueError(
                "dynamic_functions.live_strategy is enabled but no strategy is set: call "
                "set_palette_strategy(strategy) before the first generation (HMR "
                "skipped the step)"
            )
        self._live_generation += 1
        self._live_fitness_history.append(float(best_fitness))
        update = getattr(self._palette_strategy, 'post_generation_update', None)
        if callable(update):
            update(generation=self._live_generation, best_fitness=float(best_fitness),
                   fitness_history=self._live_fitness_history)
        new_palette = [int(i) for i in self._palette_strategy.get_active_palette()]
        current = [] if self.df_palette is None else [int(i) for i in np.asarray(self.df_palette)]
        if new_palette == current:
            return state
        self.df_palette = jnp.asarray(new_palette, dtype=jnp.int32)
        self._check_function_palettes()
        if self.verbose:
            added = [ACTIVATION_LIST[i] for i in sorted(set(new_palette) - set(current))]
            print(
                f"[Live Strategy] Generation {self._live_generation}: palette {new_palette} (added "
                f"{added})"
            )
        if len(new_palette) > len(current):
            state = self._perturb_activation_outputs(state, len(current), len(new_palette))
        return state

    def _perturb_activation_outputs(
        self, state: Any, old_palette_size: int, new_palette_size: int
    ) -> Any:
        """Push a share of the population toward the bins a grown palette added (HMR's name, its
        defect fixed).

        HMR added perturbation_strength to column 0 of every node of the chosen genomes:
        TensorNEAT's node key, not the bias, so the activation channel did not move and the genomes'
        node keys became fractional. It also split state.randkey to choose them. Here the strength
        goes to the bias (column 1) of the activation output node alone, and the genomes are drawn
        from a key folded from state.randkey, which stays as it is.

        Args:
            state: The state the tell returned.
            old_palette_size: The palette's length before the step (HMR's signature; unused).
            new_palette_size: Its length after (unused).

        Returns:
            The state with the chosen genomes' activation output bias moved.
        """
        rate = self._live_strategy['perturbation_rate']
        decay = self._live_strategy['perturbation_decay']
        if decay > 0 and self._live_generation > 0:
            rate *= max(0.0, 1.0 - decay * self._live_generation / 100.0)
        if rate <= 0:
            return state
        nodes = state.pop_nodes
        pop_size = nodes.shape[0]
        count = max(1, int(pop_size * rate))
        chosen = jax.random.choice(
            jax.random.fold_in(state.randkey, LIVE_STRATEGY_KEY),
            pop_size,
            shape=(count,),
            replace=False,
        )
        output_key = int(
            np.asarray(self.neat_algo.genome.output_idx)[self._cppn_output_indices['activation']]
        )
        selected = jnp.zeros(pop_size, dtype=bool).at[chosen].set(True)
        # Column 0 holds the node key (NaN in unused rows), column 1 the bias
        rows = selected[:, None] & (nodes[:, :, 0] == output_key)
        nodes = nodes.at[:, :, 1].add(
            jnp.where(rows, self._live_strategy['perturbation_strength'], 0.0)
        )
        if self.verbose:
            print(f"[Live Strategy] Perturbed {count}/{pop_size} genomes (rate {rate:.2%})")
        return state.update(pop_nodes=nodes)

    def _activation_decode_palette(self) -> Tuple[Optional[jnp.ndarray], int]:
        """What the per-node CPPN decode maps a channel into, and how many bins it uses.

        The palette and its size, or the leading ACTIVATION_LIST entries without one; under palette
        evolution's stable decode the snap table over the whole list (palette_snap_table). A palette
        that palette evolution owns must not be changed by a runner.

        Raises:
            ValueError: palette evolution is on and df_palette is not the palette it set.
        """
        evolution = getattr(self, 'df_palette_evolution', None)
        if evolution is not None:
            if not np.array_equal(
                np.asarray(self.df_palette), getattr(self, '_palette_owned', None)
            ):
                raise ValueError(
                    "dynamic_functions.palette_evolution owns df_palette, and it was changed from "
                    "outside; set one or the other"
                )
            if evolution['decode'] == 'stable':
                return jnp.asarray(palette_snap_table(self._palette_mask)), len(ACTIVATION_LIST)
        if self.df_palette is not None:
            return self.df_palette, len(self.df_palette)
        return None, self.df_num_activations

    def _palette_evolution_step(
        self, fitnesses: jnp.ndarray, custom_metrics: Dict[str, Any]
    ) -> None:
        """HMR's palette evolution, once per generation (a random walk: the comment's 'adopt only
        improving mutations' was never implemented, and is not here either).

        After ``stagnation_generations`` generations without a new run-wide best fitness, every
        function's bit flips with probability ``mutation_rate`` (kept if fewer than ``min_active``
        remain); the new palette applies from the next generation. The best comes from this
        generation's fitnesses (HMR's came from the metrics, 0.0 under skip_metrics, so it then
        mutated every five generations). Metrics: ``palette`` (the palette this generation used),
        ``palette_names``, ``palette_changed``, and ``next_palette`` when it changed (HMR recorded
        the next palette under ``palette``, only on a change).
        """
        evolution = getattr(self, 'df_palette_evolution', None)
        if evolution is None:
            return
        used = [int(i) for i in np.asarray(self.df_palette)]
        custom_metrics['palette'] = used
        custom_metrics['palette_names'] = [ACTIVATION_LIST[i] for i in used]
        values = np.asarray(fitnesses, dtype=np.float64)
        best = float(np.max(values[np.isfinite(values)])) if np.isfinite(values).any() else -np.inf
        if self._palette_best is None:
            # HMR's first call: the first generation's best, and it already counts as a stagnant
            # generation
            self._palette_best = best
        if best > self._palette_best:
            self._palette_best = best
            self._palette_stagnation = 0
        else:
            self._palette_stagnation += 1
        changed = False
        if self._palette_stagnation >= evolution['stagnation_generations']:
            self._palette_rng_key, key = jax.random.split(self._palette_rng_key)
            mask = mutate_palette_mask(
                key, self._palette_mask, evolution['mutation_rate'], evolution['min_active']
            )
            changed = not np.array_equal(mask, self._palette_mask)
            self._palette_mask = mask
            self._palette_stagnation = 0
            self.df_palette = jnp.asarray(np.flatnonzero(mask), dtype=jnp.int32)
            self._palette_owned = np.asarray(self.df_palette)
        custom_metrics['palette_changed'] = changed
        if changed:
            custom_metrics['next_palette'] = [int(i) for i in np.asarray(self.df_palette)]

    def _refuse_palette_evolution(self, path: str) -> None:
        """Palette evolution steps in the per-generation loop; a path that evaluates without it
        would keep the initial palette.

        Raises:
            ValueError: palette evolution is on.
        """
        if getattr(self, 'df_palette_evolution', None) is not None:
            raise ValueError(
                "dynamic_functions.palette_evolution steps in the per-generation loop, which "
                f"{path} does not run: the palette would never change"
            )
