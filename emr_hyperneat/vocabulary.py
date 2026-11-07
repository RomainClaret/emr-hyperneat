"""What a node can compute: the activation and aggregation functions, the neurotransmitter vectors
and the node rule of true neuromodulation, for a program that draws or checks networks without
evolving them.

Every name is the engine's own object, not a copy (``vocabulary.ACTIVATION_LIST is
emr_hyperneat.emrhyperneat.ACTIVATION_LIST``): the engine recognizes some of these functions by
identity, and a function index means the same entry here and in a configuration's palette. Nothing
here needs TensorNEAT or the configuration schemas, and importing it loads neither.

Functions:
    ``ACTIVATION_FUNCTIONS`` and ``AGGREGATION_FUNCTIONS`` map names to functions.
    ``ACTIVATION_LIST`` and ``AGGREGATION_LIST`` fix the order a function index refers to, in a
    palette and in a CPPN's function outputs; ``PALETTE_CONFIGS`` names palettes of that order.
    ``continuous_to_index`` and ``continuous_to_palette_index`` turn a CPPN output into an index
    as the engine does, by one of ``INDEX_DECODES``.

True neuromodulation (level 4):
    ``derive_receptor_from_weight`` gives receptor densities from a weight.
    ``true_neuromodulation_node_fn`` applies a node rule (``NEUROMODULATION_NODE_RULES``, in one
    of ``MODULATION_MODES``) to an activation function; ``true_neuromodulation_node_rule`` is the
    same rule with its operands kept apart, the form the engine settles through.
    ``invert_outputs`` applies the fourth neurotransmitter channel to the outputs.
    ``NT_TASK_PRESETS``, ``get_nt_for_task`` and the ``NT_PRESETS`` tables are the
    neurotransmitter vectors the engine gives tasks.

`emr_hyperneat.substrate` puts these together as the generation loop does
(`emr_hyperneat.substrate.substrate_neuromodulation`, `emr_hyperneat.substrate.substrate_forward`).
"""
from emr_hyperneat.engine.functions import (
    ACTIVATION_FUNCTIONS,
    ACTIVATION_LIST,
    AGGREGATION_FUNCTIONS,
    AGGREGATION_LIST,
    INDEX_DECODES,
    NUM_ACTIVATIONS,
    PALETTE_CONFIGS,
    continuous_to_index,
    continuous_to_palette_index,
)
from emr_hyperneat.engine.neuromodulation import (
    MODULATION_MODES,
    NEUROMODULATION_NODE_RULES,
    NT_PRESETS,
    NT_PRESETS_2,
    NT_PRESETS_3_FIBONACCI,
    NT_PRESETS_4,
    NT_PRESETS_5,
    NT_PRESETS_6,
    NT_TASK_PRESETS,
    derive_receptor_from_weight,
    get_nt_for_task,
    invert_outputs,
    true_neuromodulation_node_fn,
    true_neuromodulation_node_rule,
)

__all__ = [
    "ACTIVATION_FUNCTIONS",
    "ACTIVATION_LIST",
    "AGGREGATION_FUNCTIONS",
    "AGGREGATION_LIST",
    "INDEX_DECODES",
    "MODULATION_MODES",
    "NEUROMODULATION_NODE_RULES",
    "NT_PRESETS",
    "NT_PRESETS_2",
    "NT_PRESETS_3_FIBONACCI",
    "NT_PRESETS_4",
    "NT_PRESETS_5",
    "NT_PRESETS_6",
    "NT_TASK_PRESETS",
    "NUM_ACTIVATIONS",
    "PALETTE_CONFIGS",
    "continuous_to_index",
    "continuous_to_palette_index",
    "derive_receptor_from_weight",
    "get_nt_for_task",
    "invert_outputs",
    "true_neuromodulation_node_fn",
    "true_neuromodulation_node_rule",
]
