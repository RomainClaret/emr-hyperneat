"""Multi-task evaluation: its configuration and metrics records, the fitness of each task, branch gating
and the orthogonality of a population's task outputs."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task evolution with neuromodulation.

    Enables multi-task evolution where:
    - NEAT selection uses aggregated fitness across all tasks
    - Each task is evaluated with its task-specific NT vector
    - CPPN evolution happens once per generation (not per task)

    Attributes:
        enabled: Whether multi-task mode is active
        num_tasks: Number of tasks to evaluate
        task_names: Optional list of task names (for NT vector lookup)
        nt_per_task: Optional custom NT vectors per task
        fitness_aggregation: How to combine task fitnesses
            - 'mean': Average fitness across tasks (default)
            - 'min': Minimum fitness (hardest task matters most)
            - 'weighted': Weighted average using task_weights
            - 'product': Geometric mean
            - 'softmin': Soft minimum
            - 'harmonic': Harmonic mean
        task_weights: Optional weights for 'weighted' aggregation
        joint_evolution: If True, evolve on all tasks jointly (default True)
        orthogonality_bonus: Must stay 0.0: a non-zero value raises (one value added to every genome)
        specialization_bonus: Must stay 0.0: a non-zero value raises (one value added to every genome)
        hidden_activation: Global hidden layer activation function name (default: 'tanh')
            Use this when all tasks should use the same activation function.
            Available: 'tanh', 'sigmoid', 'relu', 'sin', 'gauss', 'identity',
                      'lelu', 'burst', 'resonator', etc. (see ACTIVATION_FUNCTIONS)
        per_task_activation: Optional per-task activation functions (overrides hidden_activation)
            Dict mapping task name to activation function name.
            Example: {'xor': 'sin', 'and': 'tanh', 'or': 'tanh'}
    """
    enabled: bool = False
    num_tasks: int = 2
    task_names: Optional[List[str]] = None
    nt_per_task: Optional[List[Any]] = None
    fitness_aggregation: str = 'mean'
    task_weights: Optional[List[float]] = None
    joint_evolution: bool = True
    orthogonality_bonus: float = 0.0
    specialization_bonus: float = 0.0
    # Dynamic activation function support for multi-task neuromodulation
    hidden_activation: str = 'tanh'  # Global activation for all tasks
    per_task_activation: Optional[Dict[str, str]] = None  # Per-task activation override

    # Extended fields (ported from neuromodulation_functions.py)
    # These are opt-in features that preserve default behavior when not set

    # Alternative task specification (Dict mapping task_name -> (inputs, targets))
    tasks: Optional[Dict[str, Tuple[Any, Any]]] = None

    # Fitness mode for evaluation
    fitness_mode: str = 'mse'  # MULTITASK_FITNESS_MODES ('ce' and unknown names raise)

    # HMR's modulation_penalty, a bonus despite its name: penalty times the variance over tasks of
    # each genome's mean modulation over its active positions (0.0 = off)
    modulation_penalty: float = 0.0

    # Softmin parameters (for 'softmin' aggregation)
    softmin_temperature: float = 0.1  # Lower = sharper (more like true min)

    # Generalist bonus mechanisms
    generalist_bonus_type: str = 'none'  # 'none', 'min_bonus', 'variance_penalty', 'threshold_bonus'
    generalist_bonus_weight: float = 0.0  # Weight for the bonus/penalty
    generalist_threshold: float = 0.9  # Threshold for 'threshold_bonus' type

    # Modulation mechanism control
    modulation_mode: str = 'full'  # 'full', 'gating_only', 'gain_bias_only'
    modulation_strength_override: Optional[float] = None  # Override default if set

    # Specialization bonus (confusion matrix gap)
    specialization_bonus_weight: float = 0.0  # Weight for NT-task alignment bonus
    confusion_eval_frequency: int = 0  # Compute confusion matrix every N generations (0 = disabled)

    # Subspace orthogonality bonus (Liu & Wang 2024 mechanism)
    orthogonality_bonus_weight: float = 0.0  # HMR's per-genome bonus; 0.0 = off
    orthogonality_metric: str = 'cosine_mean'  # 'cosine_mean', 'cosine_max', 'correlation'

    # Dendritic branch-specific gating (Liu & Wang 2024 SST mechanism)
    branch_gating_mode: str = 'none'  # 'none', 'spatial', 'hierarchical'

    # Two-Module Architecture (Liu & Wang 2024 PFC/Sensorimotor separation)
    two_module_mode: str = 'none'  # 'none', 'parallel', 'sequential'

    # HMR's names (its runners build MultiTaskConfig(tasks=..., aggregation=..., weights=...,
    # nt_vectors=...) and pass it to run_generation_multitask): aggregation is fitness_aggregation,
    # weights and nt_vectors are per task name (task_weights and nt_per_task are lists).
    aggregation: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    nt_vectors: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.aggregation is not None:
            if self.fitness_aggregation not in ('mean', self.aggregation):
                raise ValueError(
                    f"MultiTaskConfig: aggregation={self.aggregation!r} and "
                    f"fitness_aggregation={self.fitness_aggregation!r} name one setting"
                )
            self.fitness_aggregation = self.aggregation


@dataclass
class MultiTaskMetrics:
    """Metrics from multi-task evolution with per-task breakdown, keyed by task name.

    What run_generation_multitask returns for HMR's call form (a MultiTaskConfig with tasks by
    name). The aggregated fitness includes any bonus; the best generalist is the genome of the
    highest aggregated fitness.
    """
    aggregated_fitness: float = 0.0
    mean_aggregated_fitness: float = 0.0
    per_task_best: Dict[str, float] = field(default_factory=dict)
    per_task_mean: Dict[str, float] = field(default_factory=dict)
    best_generalist_per_task: Dict[str, float] = field(default_factory=dict)
    best_generalist_idx: int = 0
    generation: int = 0


#: The fitness modes of the multi-task evaluators (compute_fitness).
MULTITASK_FITNESS_MODES = ('mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy', 'accuracy_confident')
#: Accepted values of the multi-task settings (create_config raises on any other).
MULTITASK_AGGREGATIONS = ('mean', 'min', 'weighted', 'product', 'softmin', 'harmonic')


GENERALIST_BONUS_TYPES = ('none', 'min_bonus', 'variance_penalty', 'threshold_bonus')
TWO_MODULE_MODES = ('none', 'parallel', 'sequential')
#: multitask.branch_gating_mode (compute_modulation_with_branch_gating).
BRANCH_GATING_MODES = ('none', 'spatial', 'hierarchical')
#: multitask.orthogonality_metric (compute_population_orthogonality).
ORTHOGONALITY_METRICS = ('cosine_mean', 'cosine_max', 'correlation')


def refuse_anti_branch_gating(node_rule: str, branch_mode: str) -> None:
    """Refuse multi-task branch gating under neuromodulation.node_rule 'anti': the routing gives
    each region one receptor channel's modulation, which would replace the rule's own over every
    channel (HMR ANTI had no routing).

    Raises:
        ValueError: node_rule 'anti' with a branch mode other than 'none'.
    """
    if node_rule == 'anti' and branch_mode != 'none':
        raise ValueError(
            f"neuromodulation.node_rule 'anti' computes the modulation over every channel; "
            f"multitask.branch_gating_mode {branch_mode!r} would replace it"
        )


def check_branch_gating_and_penalty(
    branch_mode: Any,
    penalty: Any,
    two_module_mode: str = 'none',
    num_nt: Optional[int] = None,
    num_tasks: Optional[int] = None,
) -> None:
    """Refuse multitask.branch_gating_mode and multitask.modulation_penalty settings that cannot
    apply.

    Args:
        branch_mode: multitask.branch_gating_mode.
        penalty: multitask.modulation_penalty.
        two_module_mode: multitask.two_module_mode.
        num_nt: the NT types of the neuromodulation, when known.
        num_tasks: the number of tasks, when known.

    Raises:
        ValueError: an unknown branch mode; a branch mode with two_module_mode (both set the
            modulation by region; HMR evaluated the two modules and ignored the branch mode) or with
            fewer than three NT types (the routing reads channels 0 to 2); a penalty that is not a
            number at least 0, or above 0 with fewer than two tasks (a variance over one task is 0).
    """
    if branch_mode not in BRANCH_GATING_MODES:
        raise ValueError(
            f"multitask.branch_gating_mode must be one of {BRANCH_GATING_MODES}, got "
            f"{branch_mode!r}"
        )
    if branch_mode != 'none':
        if two_module_mode != 'none':
            raise ValueError(
                "multitask.branch_gating_mode and two_module_mode both set the modulation by "
                "region; HMR evaluated the two modules and ignored the branch mode, so set one of "
                "them"
            )
        if num_nt is not None and num_nt < 3:
            raise ValueError(
                "multitask.branch_gating_mode routes receptor channels 0 to 2; the "
                "neuromodulation has "
                f"{num_nt} NT types"
            )
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or penalty < 0:
        raise ValueError(
            f"multitask.modulation_penalty must be a number at least 0, got {penalty!r}"
        )
    if penalty > 0 and num_tasks is not None and num_tasks < 2:
        raise ValueError(
            "multitask.modulation_penalty rewards the variance of a genome's modulation over "
            "tasks: it needs two or more tasks"
        )


def compute_fitness(outputs: jnp.ndarray, targets: jnp.ndarray, mode: str = 'mse') -> float:
    """The multi-task evaluators' fitness over flattened outputs and targets: 'mse', 'accuracy', 'acc_mse', 'hybrid',
    'bce', 'soft_accuracy' and 'accuracy_confident' (accuracy plus 0.1 of the outputs' mean distance from 0.5, scaled
    to [0, 1]); MULTITASK_FITNESS_MODES.

    Raises:
        ValueError: 'ce', categorical cross-entropy over softmax outputs: these evaluators score sigmoid outputs of
            Boolean tasks, where it ranks a constant output above a solver (a constant 0.9999 scores 0.99995, a
            0.99/0.01 XOR solver 0.99499); and an unknown mode.
    """
    if mode == 'ce':
        raise ValueError("fitness mode 'ce' is categorical cross-entropy over softmax outputs; the "
                         "multi-task evaluators score sigmoid outputs, where it ranks a constant "
                         "output above a solver: use 'bce'")
    outputs = outputs.flatten()
    targets = targets.flatten()

    if mode == 'mse':
        mse = jnp.mean((outputs - targets) ** 2)
        return 1.0 - mse
    elif mode == 'accuracy':
        predictions = (outputs > 0.5).astype(jnp.float32)
        return jnp.mean(predictions == targets)
    elif mode == 'acc_mse':
        predictions = (outputs > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(predictions == targets)
        mse = jnp.mean((outputs - targets) ** 2)
        return accuracy + 0.01 * (1.0 - mse)
    elif mode == 'hybrid':
        predictions = (outputs > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(predictions == targets)
        mse = jnp.mean((outputs - targets) ** 2)
        return 0.8 * accuracy + 0.2 * (1.0 - mse)
    elif mode == 'bce':
        eps = 1e-7
        outputs_safe = jnp.clip(outputs, eps, 1.0 - eps)
        bce = -jnp.mean(
            targets * jnp.log(outputs_safe) +
            (1.0 - targets) * jnp.log(1.0 - outputs_safe)
        )
        return 1.0 / (1.0 + bce)
    elif mode == 'soft_accuracy':
        T = 10.0
        soft_pred = jax.nn.sigmoid((outputs - 0.5) * T)
        return jnp.mean(soft_pred * targets + (1.0 - soft_pred) * (1.0 - targets))
    elif mode == 'accuracy_confident':
        predictions = (outputs > 0.5).astype(jnp.float32)
        return jnp.mean(predictions == targets) + 0.1 * jnp.mean(jnp.abs(outputs - 0.5) * 2.0)
    raise ValueError(f"unknown fitness mode {mode!r}; the multi-task modes are {MULTITASK_FITNESS_MODES}")


def compute_modulation_with_branch_gating(
    receptor_densities_single: jnp.ndarray,
    neurotransmitter: jnp.ndarray,
    all_positions: jnp.ndarray,
    branch_mode: str = 'none',
) -> jnp.ndarray:
    """The modulation each position receives when its region selects the receptor channel
    (multitask.branch_gating_mode, HMR's dendritic branch gating after Liu and Wang 2024; the same
    function as HMR's compute_branch_specific_modulation).

    'none': every position receptors[:, :3] @ nt[:3], the usual modulation. 'spatial', by quadrant
    of the grid: x < 0 and y < 0 reads channel 0 alone (receptors[:, 0] * nt[0]), x >= 0 and y < 0
    channel 1, x < 0 and y >= 0 channel 2, x >= 0 and y >= 0 all three. 'hierarchical', by height:
    y < -0.33 channel 0, -0.33 <= y < 0.33 channel 1, y >= 0.33 channel 2. Both need three channels
    or more.

    Args:
        receptor_densities_single: (positions, num_nt) one genome's receptors.
        neurotransmitter: (num_nt,) the task's vector.
        all_positions: (positions, 2) the positions' grid coordinates (x, y), in the receptors'
            order.
        branch_mode: 'none', 'spatial' or 'hierarchical'.

    Returns:
        (positions,) modulation.
    """
    if branch_mode == 'none':
        return receptor_densities_single[:, :3] @ neurotransmitter[:3]

    elif branch_mode == 'spatial':
        x = all_positions[:, 0]
        y = all_positions[:, 1]
        modulation = jnp.zeros(receptor_densities_single.shape[0])

        mask_0 = (x < 0.0) & (y < 0.0)
        mod_0 = receptor_densities_single[:, 0] * neurotransmitter[0]
        modulation = jnp.where(mask_0, mod_0, modulation)

        mask_1 = (x >= 0.0) & (y < 0.0)
        mod_1 = receptor_densities_single[:, 1] * neurotransmitter[1]
        modulation = jnp.where(mask_1, mod_1, modulation)

        mask_2 = (x < 0.0) & (y >= 0.0)
        mod_2 = receptor_densities_single[:, 2] * neurotransmitter[2]
        modulation = jnp.where(mask_2, mod_2, modulation)

        mask_3 = (x >= 0.0) & (y >= 0.0)
        mod_3 = receptor_densities_single[:, :3] @ neurotransmitter[:3]
        modulation = jnp.where(mask_3, mod_3, modulation)

        return modulation

    elif branch_mode == 'hierarchical':
        y = all_positions[:, 1]
        modulation = jnp.zeros(receptor_densities_single.shape[0])

        mask_1 = y < -0.33
        mod_1 = receptor_densities_single[:, 0] * neurotransmitter[0]
        modulation = jnp.where(mask_1, mod_1, modulation)

        mask_2 = (y >= -0.33) & (y < 0.33)
        mod_2 = receptor_densities_single[:, 1] * neurotransmitter[1]
        modulation = jnp.where(mask_2, mod_2, modulation)

        mask_3 = y >= 0.33
        mod_3 = receptor_densities_single[:, 2] * neurotransmitter[2]
        modulation = jnp.where(mask_3, mod_3, modulation)

        return modulation

    else:
        raise ValueError(f"Unknown branch_gating_mode: {branch_mode}")


# Orthogonality Metrics (Liu & Wang 2024 Mechanism)
# Functions for computing subspace orthogonality bonuses to encourage
# different tasks to occupy orthogonal activation subspaces.

def compute_population_orthogonality(
    per_task_hidden: Dict[str, jnp.ndarray],
    metric: str = 'cosine_mean',
) -> jnp.ndarray:
    """Compute orthogonality bonus for entire population.

    Vectorized version that computes orthogonality bonuses for all individuals
    in the population simultaneously.

    Args:
        per_task_hidden: Dict mapping task_name -> hidden activations array
            Each array has shape (pop_size, num_cases, total_positions)
        metric: Method to compute orthogonality
            - 'cosine_mean': Mean absolute cosine similarity (default, recommended)
            - 'cosine_max': Maximum absolute cosine similarity (strictest)
            - 'correlation': Mean absolute Pearson correlation

    Returns:
        Array of orthogonality bonuses, shape (pop_size,)
    """
    task_names = list(per_task_hidden.keys())
    num_tasks = len(task_names)

    if num_tasks < 2:
        # Single task - return perfect orthogonality for everyone
        first_hidden = per_task_hidden[task_names[0]]
        pop_size = first_hidden.shape[0]
        return jnp.ones(pop_size)

    # Stack hidden activations: (num_tasks, pop_size, num_cases, total_positions)
    hidden_stack = jnp.stack([per_task_hidden[t] for t in task_names], axis=0)

    # Mean over cases -> task fingerprints: (num_tasks, pop_size, total_positions)
    mean_activations = jnp.mean(hidden_stack, axis=2)

    # Transpose for easier processing: (pop_size, num_tasks, total_positions)
    mean_activations = jnp.transpose(mean_activations, (1, 0, 2))

    # Normalize per-task fingerprints to unit vectors
    norms = jnp.linalg.norm(mean_activations, axis=2, keepdims=True) + 1e-8
    norm_activations = mean_activations / norms

    # Cosine similarity matrices per individual: (pop_size, num_tasks, num_tasks)
    # sim_matrix[p, i, j] = dot(task_i, task_j) for individual p
    sim_matrices = jnp.einsum('ptn,pmn->ptm', norm_activations, norm_activations)

    # Off-diagonal mask
    mask = 1 - jnp.eye(num_tasks)

    if metric == 'cosine_mean':
        off_diag = jnp.abs(sim_matrices) * mask  # (pop_size, num_tasks, num_tasks)
        mean_sim = jnp.sum(off_diag, axis=(1, 2)) / (num_tasks * (num_tasks - 1))
        return 1.0 - jnp.clip(mean_sim, 0.0, 1.0)

    elif metric == 'cosine_max':
        off_diag = jnp.abs(sim_matrices) * mask
        max_sim = jnp.max(off_diag, axis=(1, 2))
        return 1.0 - jnp.clip(max_sim, 0.0, 1.0)

    elif metric == 'correlation':
        # Center activations per individual per task
        centered = mean_activations - jnp.mean(mean_activations, axis=2, keepdims=True)
        std_norms = jnp.linalg.norm(centered, axis=2, keepdims=True) + 1e-8
        centered_norm = centered / std_norms
        corr_matrices = jnp.einsum('ptn,pmn->ptm', centered_norm, centered_norm)
        off_diag = jnp.abs(corr_matrices) * mask
        mean_corr = jnp.sum(off_diag, axis=(1, 2)) / (num_tasks * (num_tasks - 1))
        return 1.0 - jnp.clip(mean_corr, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown orthogonality metric: {metric}")


def _refuse_per_node_in_multitask(per_node: List[str]) -> None:
    """Refuse per-node features under the multi-task evaluation, which applies one hidden activation
    and sum aggregation: at configuration (multitask.enabled) and at every run_generation_multitask
    call, a direct one without the multitask block included (0.1.0 ignored them in both places).

    Raises:
        ValueError: PER_NODE names a feature the evaluation would ignore.
    """
    if per_node:
        raise ValueError(
            "run_generation_multitask evaluates every task with one hidden activation "
            "(multitask.per_task_activation or multitask.hidden_activation) and sum "
            "aggregation, so it would ignore " + ", ".join(per_node) + ". Configure the activation "
            "through the multitask section instead.")


#: The multi-task evaluators (`EMRHyperNEAT._create_eval_population_true_neuromodulated`), by
#: the settings they close over; bounded by the configurations a process runs.
_MULTITASK_EVALUATORS: Dict[Tuple[Any, ...], Callable] = {}
