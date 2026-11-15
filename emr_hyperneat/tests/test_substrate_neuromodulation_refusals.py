"""Two refusals of the substrate API under true neuromodulation.

Node rule ``'anti'`` (the rule of the prototype's ANTI variant) inverts no output: the generation
loop refuses an explicit ``use_output_inversion`` under it and turns an unset one off, and a
recurrent substrate refuses the inversion (`SubstrateConfig.validate`). Both feed-forward forwards,
dense and compiled, refuse it too, with the configured settings or a caller-built
`SubstrateNeuromodulation`. `NeuromodulationSettings` defaults the inversion to True, so a
hand-built ``NeuromodulationSettings(node_rule='anti')`` carries it: run with the loop's own
five-channel task vectors (one-hot, the fourth entry 0 for every task but one), the dense forward
would return exactly one minus its outputs with the inversion off for four of the five tasks. The
forwards refuse it with the recurrent path's message, whatever the channel count (as the recurrent
path does), among the checks of the settings, which come before those of the arrays and the vector
as validate's do on a recurrent substrate. What the loop also refuses under ``'anti'`` stays open on
the feed-forward path: another modulation mode (the rule does not read it) and more than four
channels (the loop's multi-task mode reads them; the vector's length is checked). On a feed-forward
configuration `SubstrateConfig.validate` accepts the combination: building reads no neuromodulation
setting, `substrate_config` never returns the combination, and the forwards refuse it wherever it
would be evaluated.

The recurrent forward evaluates each genome on its own positions, zero-padded to a bucket size. A
padding position stays inert only at finite entries of the vector the node rule reads (the first
three, every one under ``'anti'``): there m = receptors @ nt = 0. An infinite entry would make it
NaN (0 * inf), which the zero W2 rows carry to every output of the genome, so the outputs would
depend on the padding (seed 42, depth 3, 20 genomes, the paper rule under 'emr', evaluated without
the refusal: entry 0 at +inf leaves the 16 padded genomes all NaN and 2 of the 4 that fill their
bucket finite, and doubling every bucket makes those 2 NaN too); a NaN entry would make m NaN at
every position, and every output of every genome NaN, padded or not. Both are refused, an entry
finite in float64 but not in float32 (1e40) among them, the precision the forward reads the vector
at, after every other check the forward makes before the computation (the inputs' conversion
included), so a call one of those refuses keeps its message. The inversion's entry (the fourth,
outside ``'anti'``) acts after the output function and is not refused there, nor an entry nothing
reads; the dense feed-forward forward evaluates every such vector, and the compiled one refuses
them with its own message.

The tests marked as controls pin what the two refusals leave open.
"""

import copy
import functools
import re
import warnings
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    GPU_TRAJECTORY_RTOL,
    XORProblem,
    create_config_with_neuromodulation,
    exact_backend,
)
from emr_hyperneat import EMRHyperNEAT, SubstrateConfig, substrate_forward, substrate_queries
from emr_hyperneat import emrhyperneat as engine
from emr_hyperneat import substrate as api

DEPTH = 3
POP = 20
SEED = 42
#: Values not finite in float32: 1e40 is finite in float64 and inf in float32
NOT_FINITE = (np.inf, -np.inf, np.nan, 1e40)
#: The anti refusal's words, shared by both paths
ANTI_INVERSION = "set output_inversion False"


def _emr(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]


def _feed_forward(**neuromodulation: Any) -> Dict[str, Any]:
    return create_config_with_neuromodulation("true_neuromodulation_4nt",
                                              recurrence_preset="feedforward", max_depth=DEPTH,
                                              population_size=POP, **neuromodulation)


def _recurrent(settling: str, **neuromodulation: Any) -> Dict[str, Any]:
    """full_recurrent at iteration level 1, the h->h cache off (the substrate API's conditions), as
    test_substrate_recurrent_neuromodulation.py configures it."""
    config = create_config_with_neuromodulation("true_neuromodulation_4nt",
                                                recurrence_preset="full_recurrent",
                                                max_depth=DEPTH, population_size=POP,
                                                **neuromodulation)
    _emr(config)["recurrence"].update(iteration_level=1, hh_cache_enabled=False,
                                      settling_rule=settling)
    if settling == "drive_once":
        _emr(config)["dynamic_functions"]["hh_activation_mode"] = "initial_only"
    return config


#: The populations, each with its configured neuromodulation as substrate_config returns it
POPULATIONS = {
    "ff_paper": lambda: _feed_forward(),
    "ff_anti": lambda: _feed_forward(node_rule="anti", use_output_inversion=False),
    "rec_paper_emr": lambda: _recurrent("emr"),
    "rec_paper_drive_once": lambda: _recurrent("drive_once"),
    "rec_anti_emr": lambda: _recurrent("emr", node_rule="anti", use_output_inversion=False),
}


@functools.lru_cache(maxsize=None)
def _population(name: str) -> Tuple[Any, np.ndarray]:
    """A population's substrate (built by population_substrates, h->h connections included) and
    XOR's four rows; shared by the tests that read it (the arrays are not modified)."""
    algo = EMRHyperNEAT()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        state = algo.initialize(algo.create_config(copy.deepcopy(POPULATIONS[name]())),
                                XORProblem(), seed=SEED)
    substrate = algo.population_substrates(state)
    return substrate, np.asarray(XORProblem().inputs, dtype=np.float32)


def _receptors(substrate: Any, channels: int) -> jnp.ndarray:
    """The loop's receptor densities of every position with ``channels`` channels (the substrate's
    own derivation, from its first input weight)."""
    weight = substrate.W1[:, 0, :] / substrate.config.max_weight
    return api.derive_receptor_from_weight(weight, "tanh", channels) * 2.0 - 1.0


def _given(substrate: Any, settings: Any, vector: Any) -> Any:
    """A caller-built SubstrateNeuromodulation with these settings, receptors with their channels,
    the substrate's own base gains, and ``vector`` as given (no conversion)."""
    configured = api.substrate_neuromodulation(substrate)
    return configured._replace(receptors=_receptors(substrate, int(settings.num_nt_types)),
                               neurotransmitter=vector, settings=settings)


def _recurrent_message(settings: Any) -> str:
    """validate's message for ``settings`` on a recurrent configuration: the recurrent path's."""
    config = SubstrateConfig(input_coords=((-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)),
                             output_coords=((0.0, 1.0),), max_depth=2, hidden_to_hidden=True,
                             neuromodulation=settings)
    with pytest.raises(ValueError) as refused:
        config.validate()
    return str(refused.value)


def _assert_same_program(actual: Any, desired: Any) -> None:
    """One compiled program on operands that differ only where nothing reads them: bit for bit on
    the CPU, to GPU_TRAJECTORY_RTOL off it (under the 'emr' rule only, whose settling a GPU repeats,
    as test_substrate_recurrent_neuromodulation.py measured)."""
    if exact_backend():
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(desired))
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(desired),
                                   rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)


# Node rule 'anti' with the inversion on, on a feed-forward substrate

@pytest.mark.parametrize("inversion", [True, 1, np.True_], ids=["true", "one", "numpy_true"])
@pytest.mark.parametrize("compiled", [False, True], ids=["dense", "compiled"])
@pytest.mark.parametrize("how", ["configured", "given", "given_without_settings"])
def test_anti_with_the_inversion_is_refused_on_a_feed_forward_substrate(how, compiled, inversion):
    """Configured, caller-built on a configuration with neuromodulation settings, and caller-built
    on one without: refused by the dense and the compiled forward, with the recurrent path's message
    (validate's on a recurrent configuration). Run with the loop's default vector, the outputs would
    equal those with the inversion off, its fourth entry being 1.0. Any true value turns the
    inversion on in both forwards, so 1 and numpy's True are refused as True is."""
    substrate, inputs = _population("ff_anti")
    settings = substrate.config.neuromodulation._replace(output_inversion=inversion)
    expected = _recurrent_message(settings)
    assert ANTI_INVERSION in expected
    if how == "configured":
        target = substrate._replace(config=substrate.config._replace(neuromodulation=settings))
        given = None
    else:
        target = substrate
        if how == "given_without_settings":
            target = substrate._replace(config=substrate.config._replace(neuromodulation=None))
        given = api.substrate_neuromodulation(substrate)._replace(settings=settings)
    with pytest.raises(ValueError) as refused:
        substrate_forward(target, inputs, neuromodulation=given, compiled=compiled)
    assert str(refused.value) == expected


@pytest.mark.parametrize("channels", [2, 3, 4, 5, 6])
def test_hand_built_anti_settings_keep_the_default_inversion_and_are_refused(channels):
    """NeuromodulationSettings(node_rule='anti') inverts by default. Refused at every channel count,
    dense and compiled, as on a recurrent substrate (the inversion reads no entry below four
    channels, so there it would change nothing; at five and six with the loop's one-hot task vectors
    it would return one minus its outputs with the inversion off); with output_inversion False the
    same call runs. Five and six channels: the loop's multi-task vectors (get_nt_for_task), each
    refused."""
    substrate, inputs = _population("ff_anti")
    settings = api.NeuromodulationSettings(node_rule="anti", num_nt_types=channels)
    assert settings.output_inversion is True
    vectors = ([engine.get_nt_for_task(task, channels) for task in ("xor", "and", "nand")]
               if channels > 4 else [api.default_neurotransmitter(channels)])
    for vector in vectors:
        for compiled in (False, True):
            with pytest.raises(ValueError, match=ANTI_INVERSION):
                substrate_forward(substrate, inputs, compiled=compiled,
                                  neuromodulation=_given(substrate, settings, vector))
            off = _given(substrate, settings._replace(output_inversion=False), vector)
            outputs = np.asarray(substrate_forward(substrate, inputs, neuromodulation=off,
                                                   compiled=compiled))
            assert outputs.shape == (POP, inputs.shape[0], 1) and np.isfinite(outputs).all()


def test_the_settings_are_checked_before_the_arrays_and_the_vector():
    """On the feed-forward path the refusal is one of the settings' checks, which run first, as
    validate's do on a recurrent substrate: with base gains that do not cover the substrate, or a
    vector too short for the rule (which reads every channel under 'anti'), the call is refused for
    'anti' with the inversion, not for the arrays or the vector. An unknown modulation mode keeps
    the feed-forward path's own message, checked before it."""
    substrate, inputs = _population("ff_anti")
    settings = substrate.config.neuromodulation._replace(output_inversion=True)
    configured = api.substrate_neuromodulation(substrate)
    expected = _recurrent_message(settings)
    for wrong in (configured._replace(settings=settings, base_gains=configured.base_gains[:1]),
                  configured._replace(settings=settings,
                                      neurotransmitter=jnp.asarray([0.5, 0.5, 0.5]))):
        for compiled in (False, True):
            with pytest.raises(ValueError) as refused:
                substrate_forward(substrate, inputs, neuromodulation=wrong, compiled=compiled)
            assert str(refused.value) == expected
    unknown = configured._replace(settings=settings._replace(modulation_mode="other"))
    for compiled in (False, True):
        with pytest.raises(ValueError, match="unknown modulation mode 'other' or node rule 'anti'"):
            substrate_forward(substrate, inputs, neuromodulation=unknown, compiled=compiled)
    # The recurrent path: validate before the arrays
    recurrent, recurrent_inputs = _population("rec_anti_emr")
    on_recurrent = api.substrate_neuromodulation(recurrent)
    with pytest.raises(ValueError) as refused:
        substrate_forward(recurrent, recurrent_inputs, neuromodulation=on_recurrent._replace(
            settings=settings, base_gains=on_recurrent.base_gains[:1]))
    assert str(refused.value) == expected


def test_control_anti_without_the_inversion_runs_as_before():
    """Control: what the anti refusal leaves open. 'anti' with the inversion off, as
    substrate_config gives it, runs in both forwards; the modulation mode is unread under 'anti'
    (another mode gives the same outputs, bit for bit, where the loop refuses it); five and six
    channels with a vector of that length run and read every entry (the loop's multi-task mode reads
    them), while the four-entry default is refused by the vector's length check."""
    substrate, inputs = _population("ff_anti")
    settings = substrate.config.neuromodulation
    assert settings.node_rule == "anti" and settings.output_inversion is False
    for compiled in (False, True):
        full = np.asarray(substrate_forward(substrate, inputs, compiled=compiled))
        assert np.isfinite(full).all()
        for mode in ("gating_only", "gain_bias_only"):
            other = api.substrate_neuromodulation(substrate)._replace(
                settings=settings._replace(modulation_mode=mode))
            np.testing.assert_array_equal(
                np.asarray(substrate_forward(substrate, inputs, neuromodulation=other,
                                             compiled=compiled)), full)
        for channels in (5, 6):
            wide = settings._replace(num_nt_types=channels)
            with pytest.raises(ValueError, match=f"reads {channels} entries"):
                substrate_forward(substrate, inputs, compiled=compiled, neuromodulation=_given(
                    substrate, wide, api.default_neurotransmitter(channels)))
            first = np.asarray(substrate_forward(substrate, inputs, compiled=compiled,
                                                 neuromodulation=_given(substrate, wide,
                                                                        jnp.full(channels, 0.5))))
            last = jnp.full(channels, 0.5).at[channels - 1].set(0.9)
            moved = np.asarray(substrate_forward(substrate, inputs, compiled=compiled,
                                                 neuromodulation=_given(substrate, wide, last)))
            assert np.isfinite(first).all() and float(np.abs(moved - first).max()) > 0.0


def test_control_validate_leaves_a_feed_forward_configuration_to_the_forwards():
    """Control, where the anti refusal sits: validate refuses 'anti' with the inversion on a
    recurrent configuration and not on a feed-forward one, whose building reads no neuromodulation
    setting (substrate_queries and build_substrate validate). substrate_config never returns the
    combination: the loop refuses an explicit inversion under 'anti' and turns an unset one off,
    feed-forward or recurrent. A hand-built configuration can carry it (the NamedTuple's default),
    and the forwards refuse it there."""
    for preset in ("feedforward", "full_recurrent"):
        unset = create_config_with_neuromodulation("true_neuromodulation_4nt",
                                                   recurrence_preset=preset, max_depth=DEPTH,
                                                   population_size=POP, node_rule="anti")
        if preset != "feedforward":
            _emr(unset)["recurrence"].update(iteration_level=1, hh_cache_enabled=False)
        algo = EMRHyperNEAT()
        algo.create_config(copy.deepcopy(unset))
        assert algo.substrate_config().neuromodulation.output_inversion is False, preset
        explicit = copy.deepcopy(unset)
        _emr(explicit)["neuromodulation"]["use_output_inversion"] = True
        with pytest.raises(ValueError, match="inverts no output"):
            EMRHyperNEAT().create_config(explicit)
    substrate, _ = _population("ff_anti")
    hand_built = substrate.config._replace(
        neuromodulation=api.NeuromodulationSettings(node_rule="anti"))
    assert hand_built.validate() is hand_built
    assert substrate_queries(hand_built).stacked().shape == substrate_queries(
        substrate.config).stacked().shape
    with pytest.raises(ValueError, match=ANTI_INVERSION):
        hand_built._replace(hidden_to_hidden=True).validate()


# A non-finite entry the node rule reads, on a recurrent substrate

def _entries_read(settings: Any) -> int:
    """The entries m reads: every one under 'anti', otherwise the first three."""
    channels = int(settings.num_nt_types)
    return channels if settings.node_rule == "anti" else min(3, channels)


@pytest.mark.parametrize("name", ["rec_paper_emr", "rec_paper_drive_once", "rec_anti_emr"])
def test_a_recurrent_vector_entry_the_node_rule_reads_must_be_finite_in_float32(name):
    """Each entry m reads (the first three; every one, the fourth included, under 'anti') at +-inf,
    NaN and 1e40 is refused, through substrate_neuromodulation (which converts the vector to
    float32) and as a caller's float64 vector, with and without return_states, under both settling
    rules; the check itself refuses the float64 vector as given, which is what the forward hands it
    under JAX_ENABLE_X64 (1e40 is finite in float64, and the forward reads it in float32).
    Evaluated, an infinite entry (1e40 among them) would make the outputs depend on the bucket
    padding, and a NaN one would make every output NaN."""
    substrate, inputs = _population(name)
    configured = api.substrate_neuromodulation(substrate)
    read = _entries_read(configured.settings)
    message = (f"the neurotransmitter vector's entries the node rule reads \\(the first {read}\\) "
               f"must be finite in float32 on a recurrent substrate")
    for entry in range(read):
        for value in NOT_FINITE:
            # A copy: under JAX_ENABLE_X64 np.asarray would return a read-only view of the vector
            vector = np.array(api.default_neurotransmitter(int(configured.settings.num_nt_types)),
                              dtype=np.float64)
            vector[entry] = value
            with warnings.catch_warnings():
                # numpy's overflow warning, casting 1e40 to float32 (here or in the forward)
                warnings.simplefilter("ignore", RuntimeWarning)
                converted = api.substrate_neuromodulation(substrate, neurotransmitter=vector)
                for given in (converted, configured._replace(neurotransmitter=vector)):
                    for return_states in (False, True):
                        with pytest.raises(ValueError, match=message):
                            substrate_forward(substrate, inputs, return_states=return_states,
                                              neuromodulation=given)
                refusal = api._neurotransmitter_entries_refusal(
                    configured.settings, vector, inversion=False, where="on a recurrent substrate")
            assert refusal is not None and re.search(message, refusal), (entry, value)


def test_fewer_channels_read_fewer_entries_on_a_recurrent_substrate():
    """With two channels the rule reads two entries: a non-finite second one is refused, a third
    one is read by nothing and may be anything (control half: it runs, bit for bit as a finite
    third entry)."""
    substrate, inputs = _population("rec_paper_emr")
    settings = substrate.config.neuromodulation._replace(num_nt_types=2)
    with pytest.raises(ValueError, match=r"node rule reads \(the first 2\)"):
        substrate_forward(substrate, inputs,
                          neuromodulation=_given(substrate, settings, jnp.asarray([0.5, np.inf])))
    finite = np.asarray(substrate_forward(substrate, inputs, neuromodulation=_given(
        substrate, settings, jnp.asarray([0.5, 0.5, 0.5]))))
    unread = np.asarray(substrate_forward(substrate, inputs, neuromodulation=_given(
        substrate, settings, jnp.asarray([0.5, 0.5, np.inf]))))
    _assert_same_program(unread, finite)


def test_control_the_inversion_entry_and_unread_entries_stay_open_on_a_recurrent_substrate(
        monkeypatch):
    """Control: what the entry refusal leaves open. Under the paper rule the inversion clips its
    entry to [0, 1] after the output function, so +inf and 1e40 give the outputs of 1.0 and -inf
    those of 0.0, bit for bit, and NaN gives NaN at every output of every genome (no genome depends
    on its padding); with the inversion off the fourth entry is unread; with six channels the fifth
    and sixth are unread; an entry finite in float32 at its largest magnitude is not refused, and
    its outputs do not depend on the padding, where m stays 0: with every bucket doubled they are
    the same, finite where they were (two programs, so within GPU_TRAJECTORY_RTOL; an Apple M4 Max
    gave the same bits)."""
    substrate, inputs = _population("rec_paper_emr")
    configured = api.substrate_neuromodulation(substrate)
    settings = configured.settings
    assert settings.node_rule == "paper" and settings.output_inversion

    def run(vector, given_settings=settings):
        with warnings.catch_warnings():
            # numpy's overflow warning, casting 1e40 to float32 in the forward
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.asarray(substrate_forward(substrate, inputs, neuromodulation=_given(
                substrate, given_settings, np.asarray(vector, dtype=np.float64))))

    one, zero = run([0.5, 0.5, 0.5, 1.0]), run([0.5, 0.5, 0.5, 0.0])
    for value, expected in ((np.inf, one), (1e40, one), (-np.inf, zero)):
        _assert_same_program(run([0.5, 0.5, 0.5, value]), expected)
    assert np.isnan(run([0.5, 0.5, 0.5, np.nan])).all()
    off = settings._replace(output_inversion=False)
    _assert_same_program(run([0.5, 0.5, 0.5, np.inf], off), run([0.5, 0.5, 0.5, 1.0], off))
    six = settings._replace(num_nt_types=6)
    _assert_same_program(run([0.5, 0.5, 0.5, 1.0, np.inf, np.nan], six),
                         run([0.5, 0.5, 0.5, 1.0, 0.5, 0.5], six))
    largest = float(np.finfo(np.float32).max)
    huge = [largest, -largest, largest, 1.0]
    as_built = run(huge)
    assert as_built.shape == (POP, inputs.shape[0], 1)
    bucket_size = api._bucket_size_for
    monkeypatch.setattr(api, "_bucket_size_for", lambda count: 2 * bucket_size(count))
    doubled = run(huge)
    np.testing.assert_array_equal(np.isfinite(doubled), np.isfinite(as_built))
    np.testing.assert_allclose(doubled, as_built, rtol=GPU_TRAJECTORY_RTOL, atol=1e-6)


def test_control_what_the_recurrent_path_refused_keeps_its_message():
    """Control: the entry check comes after every other check the recurrent forward makes before
    the computation, so a call one of those refuses keeps its message whatever the vector's entries
    hold: a vector too short for the entries the rule and the inversion read (its length is checked
    first), and inputs that do not convert to a float32 array (converted before the entries are
    checked)."""
    substrate, inputs = _population("rec_paper_emr")
    configured = api.substrate_neuromodulation(substrate)
    short = configured._replace(neurotransmitter=jnp.asarray([np.inf, 0.5, 0.5]))
    with pytest.raises(ValueError, match="reads 4 entries"):
        substrate_forward(substrate, inputs, neuromodulation=short)
    infinite = configured._replace(neurotransmitter=jnp.asarray([np.inf, 0.5, 0.5, 1.0]))
    ragged = [list(inputs[0]), list(inputs[1][:1])]
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        substrate_forward(substrate, ragged, neuromodulation=infinite)


def test_control_the_feed_forward_forwards_keep_their_vector_checks():
    """Control: the entry refusal is the recurrent path's. The dense feed-forward forward evaluates
    a non-finite entry the rule reads (every output NaN), and the compiled one refuses it with its
    own message, which counts the inversion's entry with the node rule's (the two forwards share
    the check, not the message)."""
    substrate, inputs = _population("ff_paper")
    vector = np.asarray([np.inf, 0.5, 0.5, 1.0], dtype=np.float64)
    given = api.substrate_neuromodulation(substrate)._replace(neurotransmitter=vector)
    dense = np.asarray(substrate_forward(substrate, inputs, neuromodulation=given))
    assert dense.shape == (POP, inputs.shape[0], 1) and np.isnan(dense).all()
    with pytest.raises(ValueError) as refused:
        substrate_forward(substrate, inputs, neuromodulation=given, compiled=True)
    assert str(refused.value) == (
        "the neurotransmitter vector's entries the node rule and the inversion read (the first 4) "
        "must be finite in float32 with compiled=True (the forward reads them in float32), got "
        "[inf, 0.5, 0.5, 1.0]")
