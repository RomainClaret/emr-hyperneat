"""How the EMR versions of this paper's runners reproduce the HMR engine's runs.

From release 0.2.0 the benchmark_bio_* runners of this folder run EMR-HyperNEAT; their published
runs used the HMR engine, which release 0.1.0 ships. The HMR engine applied two behaviors that
EMR keeps as options: its level-0 variance rule (`variance_rule: legacy`, one variance over level
0's own four weights deciding the whole level) and an extra split of the random key each generation
(`extra_randkey_split: True`). USE_HMR_RULE_AND_SPLIT pins both in the configuration of every
runner listed above, as the published runs had them; set it to False to run EMR's defaults (the
canonical rule, no extra split).
"""

from typing import Any, Dict

#: Whether the runners pin HMR's level-0 rule and key split (True, as the published runs had them;
#: False runs EMR's defaults).
USE_HMR_RULE_AND_SPLIT = True


def hmr_rule_and_split() -> Dict[str, Any]:
    """The emr_hyperneat keys of HMR's level-0 rule and key split, or none when switched off."""
    if USE_HMR_RULE_AND_SPLIT:
        return {'variance_rule': 'legacy', 'extra_randkey_split': True}
    return {}
