"""Palette Evolution Strategies for HMR-HyperNEAT.

This module contains different palette evolution strategies to compare
their effectiveness at discovering useful activation functions (like sin
for parity problems).

Strategies:
1. Baseline - Current implementation (10% uniform, stagnation-triggered)
2. Fitness-Guided - Track elite usage, bias mutations accordingly
3. Asymmetric+Sticky - Higher activation rate, sticky discoveries
4. Sin Default - Include sin in starting palette (no evolution)
5. Separate Genome - NEAT-style palette evolution with speciation

Bio-Inspired Strategies:
6. Exploration Bonus - Fix separate_genome with novelty/curiosity bonuses
7. Neuromodulated - Dynamic rates via DA/ACh/NE neuromodulation
8. Hebbian - Co-occurrence learning "fire together, stay together"
9. Critical Period - Developmental windows with phase-specific plasticity

Hybrid Strategies:
10. NeuroHebbian - Combines neuromodulation with Hebbian co-occurrence
11. CriticalSticky - Critical periods with sticky oscillatory protection
12. CriticalHebbian - Critical periods with LEARNED protection (no hard-coding)

Extended Bio-Inspired Strategies:
13. HomeostaticHebbian - CriticalHebbian with synaptic scaling (prevent saturation)
14. DecayingHebbian - CriticalHebbian with use-it-or-lose-it decay
15. Metaplastic - CriticalHebbian with adaptive learning rates (BCM-inspired)

Advanced Bio-Inspired Strategies:
16. STDP - Spike-timing-dependent plasticity (temporal order matters)
17. CompetitiveHebbian - Zero-sum resource competition (lateral inhibition)
18. NeuralDarwinism - Selective stabilization with cooperation/antagonism

Memory Consolidation Strategies:
19. ConsolidationWindow - Periodic consolidation "sleep" phases
20. SynapticTagging - Two-stage tagging + capture mechanism
21. MultiNeuromodulatory - Full ACh/DA/NE/5-HT system with interactions

Advanced Learning Mechanisms:
22. EligibilityTrace - Three-factor dopamine-gated eligibility trace learning
23. ComplementaryLearning - Hippocampus-cortex dual memory system
24. PredictiveCoding - Prediction-error-driven learning (surprise teaches)

Behavioral/Spatial and New Bio-Inspired:
25. IntrinsicPlasticity - Per-function threshold/gain adaptation (neuron excitability)
26. EcologicalSuccession - Developmental phases (pioneer → climax, generalist → specialist)
27. QuorumSensing - Population-level consensus (bacterial collective behavior)

Advanced Bio-Inspired Strategies:
28. DendriticComputation - Zone-based local processing before global integration
29. ClonalSelection - Immune-inspired affinity-based selection with diversity
30. PredatorPrey - Lotka-Volterra oscillations between generalist/specialist
31. MorphogenGradient - Developmental spatial fields with diffusion gradients

Population and Rhythmic Strategies:
32. AntColonyPheromone - Population stigmergy through pheromone trails
33. GeneticRegulatoryNetwork - Boolean regulatory circuits for expression control
34. CircadianRhythm - Intrinsic oscillatory gating of function availability

Protection and Resource Strategies:
35. SynapticFatigue - Temporal fatigue dynamics with use-dependent depression
36. BurstRefractory - Refractory periods after burst firing activity
37. RetrogradeSignaling - Backward credit assignment through activation chains
38. GlialModulation - Energy-based constraints with astrocyte-like support
39. AnchorInhibition - Probabilistic protection based on fitness contribution
40. ProgressivePalette - Inherit-and-expand architecture for zero forgetting

Advanced Bio-Inspired Strategies:
41. AdultNeurogenesis - Hippocampal neurogenesis with birth, maturation, survival
42. SynapticHomeostasisHypothesis - Sleep/wake consolidation cycles
43. OpioidNeuromodulation - Reward-driven explore/exploit with tolerance
44. ImmuneMemory - Adaptive immunity with memory cells and cross-reactivity
45. CriticalPeriodRefined - Multiple overlapping periods with gradual closure
46. EcosystemDynamics - Ecological succession with keystone functions

Dual Palette Strategies (Activation + Aggregation):
These strategies extend bio-inspired mechanisms to jointly evolve BOTH activation
AND aggregation function palettes with cross-domain learning.

7D. NeuromodulatedDual - DA/ACh/NE neuromodulation for both domains
8D. HebbianDual - Co-occurrence learning in both domains
9D. CriticalPeriodDual - Developmental windows for both domains
12D. CriticalHebbianDual - Learned protection for both domains
13D. HomeostaticHebbianDual - Synaptic scaling for both domains
17D. CompetitiveHebbianDual - Zero-sum competition in both domains
18D. NeuralDarwinismDual - Cooperation/antagonism for both domains
19D. ConsolidationWindowDual - Sleep/replay for both domains
20D. SynapticTaggingDual - Tag & capture for both domains
22D. EligibilityTraceDual - Three-factor learning, shared dopamine
23D. ComplementaryLearningDual - Fast/slow memory for both domains
24D. PredictiveCodingDual - Prediction error for both domains

Extended Dual Strategies (Activation + Aggregation):
14D. DecayingHebbianDual - Use-it-or-lose-it decay for both domains
15D. MetaplasticDual - BCM-inspired adaptive learning rates for both domains
16D. STDPDual - Spike-timing plasticity for both domains
21D. MultiNeuromodulatoryDual - Full 4-neurotransmitter system for both domains
25D. IntrinsicPlasticityDual - Threshold/gain adaptation for both domains
26D. EcologicalSuccessionDual - Pioneer→climax phases for both domains
28D. DendriticComputationDual - Zone-based local processing for both domains
31D. MorphogenGradientDual - Developmental gradients for both domains
27D. QuorumSensingDual - Population-level voting for both domains
29D. ClonalSelectionDual - Immune affinity selection for both domains
30D. PredatorPreyDual - Lotka-Volterra oscillations for both domains
6D. ExplorationBonusDual - Novelty/curiosity bonuses for both domains
3D. AsymmetricStickyDual - Asymmetric rates + sticky for both domains
10D. NeuroHebbianDual - Neuromodulation + Hebbian for both domains

Hybrid Strategies:
67. CircadianClonalDual - Circadian rhythms + clonal selection
68. CriticalStdpDual - Critical periods + STDP consolidation
69. ConsolidationClonalDual - Memory consolidation + clonal selection
70. CircadianCriticalDual - Circadian + critical periods

Aggregation-Focused Strategies:
72. AggHomeostasisDual - Homeostatic balance between averaging/extreme aggregations
73. AggGatedRetentionDual - Thalamic gating protects high-attribution aggregations
74. CrossDomainReinforcementDual - Neuromodulatory cross-domain affinity reinforcement
75. AggCriticalPeriodDual - Shorter aggregation critical periods (consolidate early)
76. ExtremeValueBiasDual - Explicit bias toward extreme aggregations (max/min)
77. AggStabilityLockDual - Metaplasticity locks long-active aggregations

Aggregation Discovery Strategies:
78. GRNAggDiscoveryDual - GRN with sin-extreme regulatory coupling
79. CriticalPeriodAggDual - Offset critical periods (aggregation consolidates first)
80. HebbianAggDiscoveryDual - Cross-domain 18x6 Hebbian matrix for act-agg pairs
81. SynapticTaggingAggDual - Cross-domain tag-and-capture with extreme boost
82. HomeostaticAggDiscoveryDual - Homeostatic balance with active discovery
83. STDPAggDiscoveryDual - Temporal credit for act-agg pair discovery

Hybrid Strategies (Combining Winners):
Part A (Hybrid Combinations):
84. TagHomeostaticDual - Tag-and-capture + homeostatic balance (81+82)
85. TagReinforcementDual - Tag-and-capture + cross-domain reinforcement (81+74)
86. HomeostaticReinforcementDual - Homeostatic balance + reinforcement (82+74)
87. TripleHybridDual - All three mechanisms combined (81+82+74)
88. GRNFixedDual - Fixed GRN with reduced coupling (78 parameter fix)
89. STDPTagDual - STDP temporal credit + tag-and-capture (83+81)

Part B (Bio-Inspired Extensions on Best Performer tag_homeostatic_dual):
90. NeurogenesisHybridDual - Tag+Homeostatic + adult neurogenesis (84+63)
91. ClonalHybridDual - Tag+Homeostatic + clonal selection (84+29) **BREAKTHROUGH: 100% Parity-5**
92. NeuromodHybridDual - Tag+Homeostatic + multi-neuromodulatory (84+21)

Extended Investigation (12 strategies):
Full CL sequence validation: XOR → Parity-3 → Parity-4 → Parity-5 → Parity-6

Direction A (Extend Clonal Winner):
93. ClonalPredictiveDual - Clonal + predictive coding (91+24) - 100% sin retention
94. ClonalDendriticDual - Clonal + zone-based protection (91+28) - 100% sin retention
95. ClonalRetrogradeDual - Clonal + backward credit cascades (91+37) - 100% sin retention
96. ClonalBCMDual - Clonal + BCM sliding threshold (91+15) - 100% sin retention

Direction B (Fix STDP Retention):
97. STDPDomainAwareDual - Domain shift detection, suspend LTD - 100% sin retention
98. STDPExtendedWindowDual - Longer LTP/LTD windows (8/10) - 100% sin retention
99. STDPNoTagDual - Pure STDP without tagging - 50% sin (FAILS - confirms tagging isn't issue)
100. STDPAdaptiveWindowDual - Dynamic windows adapt to fitness - 100% sin retention

Direction C (Aggregation-First Discovery): **BEST DIRECTION**
101. AggregationLedDual - Discover aggs first, then activations - 100% sin, 100% stability 🥇
102. DualIndependentHomeostasis - Separate homeostatic systems - 100% sin retention
103. CrossDomainV2Dual - Multiplicative cross-domain coupling - 100% sin, 100% stability 🥇
104. AggregationAnchorDual - Near-permanent agg anchoring - 100% sin retention

CHAMPIONS (100% sin retention + 100% palette stability):
- aggregation_led_dual (101) - Minimal palette (3 acts, 4 aggs)
- cross_domain_v2_dual (103) - Minimal palette (4 acts, 2 aggs)

Total strategies: 104 (46 single-domain + 58 dual-domain)
"""

from .base_strategy import PaletteEvolutionStrategy
from .strategy_1_baseline import BaselineStrategy
from .strategy_2_fitness_guided import FitnessGuidedStrategy
from .strategy_3_asymmetric_sticky import AsymmetricStickyStrategy
from .strategy_4_sin_in_default import SinDefaultStrategy
from .strategy_5_separate_genome import SeparateGenomeStrategy
from .strategy_6_exploration_bonus import ExplorationBonusStrategy
from .strategy_7_neuromodulated import NeuromodulatedStrategy
from .strategy_8_hebbian import HebbianStrategy
from .strategy_9_critical_period import CriticalPeriodStrategy
from .strategy_10_neuro_hebbian import NeuroHebbianStrategy
from .strategy_11_critical_sticky import CriticalStickyStrategy
from .strategy_12_critical_hebbian import CriticalHebbianStrategy
from .strategy_13_homeostatic_hebbian import HomeostaticHebbianStrategy
from .strategy_14_decaying_hebbian import DecayingHebbianStrategy
from .strategy_15_metaplastic import MetaplasticStrategy
from .strategy_16_stdp import STDPStrategy
from .strategy_17_competitive_hebbian import CompetitiveHebbianStrategy
from .strategy_18_neural_darwinism import NeuralDarwinismStrategy
from .strategy_19_consolidation_window import ConsolidationWindowStrategy
from .strategy_20_synaptic_tagging import SynapticTaggingStrategy
from .strategy_21_multi_neuromodulatory import MultiNeuromodulatoryStrategy
from .strategy_22_eligibility_trace import EligibilityTraceStrategy
from .strategy_23_complementary_learning import ComplementaryLearningStrategy
from .strategy_24_predictive_coding import PredictiveCodingStrategy
from .strategy_25_intrinsic_plasticity import IntrinsicPlasticityStrategy
from .strategy_26_ecological_succession import EcologicalSuccessionStrategy
from .strategy_27_quorum_sensing import QuorumSensingStrategy
from .strategy_28_dendritic_computation import DendriticComputationStrategy
from .strategy_29_clonal_selection import ClonalSelectionStrategy
from .strategy_29_clonal_selection_memcell import ClonalSelectionMemcellStrategy
from .strategy_30_predator_prey import PredatorPreyStrategy
from .strategy_31_morphogen_gradient import MorphogenGradientStrategy

# Strategies
from .strategy_32_ant_colony_pheromone import AntColonyPheromoneStrategy
from .strategy_33_genetic_regulatory_network import GeneticRegulatoryNetworkStrategy
from .strategy_34_circadian_rhythm import CircadianRhythmStrategy

# Strategies
from .strategy_35_synaptic_fatigue import SynapticFatigueStrategy
from .strategy_36_burst_refractory import BurstRefractoryStrategy
from .strategy_37_retrograde_signaling import RetrogradeSignalingStrategy
from .strategy_38_glial_modulation import GlialModulationStrategy
from .strategy_39_anchor_inhibition import AnchorInhibitionStrategy
from .strategy_40_progressive_palette import ProgressivePaletteStrategy

# Strategies
from .strategy_41_adult_neurogenesis import AdultNeurogenesisStrategy
from .strategy_42_synaptic_homeostasis_hypothesis import SynapticHomeostasisHypothesisStrategy
from .strategy_43_opioid_neuromodulation import OpioidNeuromodulationStrategy
from .strategy_44_immune_memory import ImmuneMemoryStrategy
from .strategy_45_critical_period_refined import CriticalPeriodRefinedStrategy
from .strategy_46_ecosystem_dynamics import EcosystemDynamicsStrategy

# Dual Palette Strategies
from .strategy_7_neuromodulated_dual import NeuromodulatedDualStrategy
from .strategy_8_hebbian_dual import HebbianDualStrategy
from .strategy_9_critical_period_dual import CriticalPeriodDualStrategy
from .strategy_12_critical_hebbian_dual import CriticalHebbianDualStrategy
from .strategy_13_homeostatic_hebbian_dual import HomeostaticHebbianDualStrategy
from .strategy_14_decaying_hebbian_dual import DecayingHebbianDualStrategy
from .strategy_15_metaplastic_dual import MetaplasticDualStrategy
from .strategy_16_stdp_dual import STDPDualStrategy
from .strategy_17_competitive_hebbian_dual import CompetitiveHebbianDualStrategy
from .strategy_18_neural_darwinism_dual import NeuralDarwinismDualStrategy
from .strategy_19_consolidation_window_dual import ConsolidationWindowDualStrategy
from .strategy_20_synaptic_tagging_dual import SynapticTaggingDualStrategy
from .strategy_21_multi_neuromodulatory_dual import MultiNeuromodulatoryDualStrategy
from .strategy_22_eligibility_trace_dual import EligibilityTraceDualStrategy
from .strategy_23_complementary_learning_dual import ComplementaryLearningDualStrategy
from .strategy_24_predictive_coding_dual import PredictiveCodingDualStrategy
from .strategy_25_intrinsic_plasticity_dual import IntrinsicPlasticityDualStrategy
from .strategy_26_ecological_succession_dual import EcologicalSuccessionDualStrategy
from .strategy_28_dendritic_computation_dual import DendriticComputationDualStrategy
from .strategy_31_morphogen_gradient_dual import MorphogenGradientDualStrategy
from .strategy_27_quorum_sensing_dual import QuorumSensingDualStrategy
from .strategy_29_clonal_selection_dual import ClonalSelectionDualStrategy
from .strategy_30_predator_prey_dual import PredatorPreyDualStrategy
from .strategy_6_exploration_bonus_dual import ExplorationBonusDualStrategy
from .strategy_3_asymmetric_sticky_dual import AsymmetricStickyDualStrategy
from .strategy_10_neuro_hebbian_dual import NeuroHebbianDualStrategy

# New Dual Strategies (Complete Conversion)
# Batch 1 - Baseline Dual
from .strategy_47_baseline_dual import BaselineDualStrategy
from .strategy_48_fitness_guided_dual import FitnessGuidedDualStrategy
from .strategy_49_sin_default_dual import SinDefaultDualStrategy
# Batch 2 - Population Dual
from .strategy_50_ant_colony_pheromone_dual import AntColonyPheromoneDualStrategy
from .strategy_51_ecosystem_dynamics_dual import EcosystemDynamicsDualStrategy
# Batch 3 - Temporal Dual
from .strategy_52_circadian_rhythm_dual import CircadianRhythmDualStrategy
from .strategy_53_burst_refractory_dual import BurstRefractoryDualStrategy
# Batch 4 - Neuromodulation Dual
from .strategy_54_synaptic_fatigue_dual import SynapticFatigueDualStrategy
from .strategy_55_retrograde_signaling_dual import RetrogradeSignalingDualStrategy
from .strategy_56_glial_modulation_dual import GlialModulationDualStrategy
from .strategy_57_opioid_neuromodulation_dual import OpioidNeuromodulationDualStrategy
# Batch 5 - Memory Dual
from .strategy_58_anchor_inhibition_dual import AnchorInhibitionDualStrategy
from .strategy_59_synaptic_homeostasis_dual import SynapticHomeostasisDualStrategy
from .strategy_60_immune_memory_dual import ImmuneMemoryDualStrategy
# Batch 6 - Developmental Dual
from .strategy_61_genetic_regulatory_network_dual import GeneticRegulatoryNetworkDualStrategy
from .strategy_62_progressive_palette_dual import ProgressivePaletteDualStrategy
from .strategy_63_adult_neurogenesis_dual import AdultNeurogenesisDualStrategy
from .strategy_64_critical_period_refined_dual import CriticalPeriodRefinedDualStrategy
# Batch 7 - Remaining Dual
from .strategy_65_separate_genome_dual import SeparateGenomeDualStrategy
from .strategy_66_critical_sticky_dual import CriticalStickyDualStrategy

# Hybrid Strategies
from .strategy_hybrid_progressive_consolidation import ProgressiveConsolidationStrategy
from .strategy_hybrid_ecosystem_neurogenesis import EcosystemNeurogenesisStrategy
from .strategy_hybrid_clonal_immune import ClonalImmuneStrategy

# Aggregation-Focused Strategies
from .strategy_72_agg_homeostasis_dual import AggHomeostasisDualStrategy
from .strategy_73_agg_gated_retention_dual import AggGatedRetentionDualStrategy
from .strategy_74_cross_domain_reinforcement_dual import CrossDomainReinforcementDualStrategy
from .strategy_75_agg_critical_period_dual import AggCriticalPeriodDualStrategy
from .strategy_76_extreme_value_bias_dual import ExtremeValueBiasDualStrategy
from .strategy_77_agg_stability_lock_dual import AggStabilityLockDualStrategy

# Aggregation Discovery Strategies
from .strategy_78_grn_agg_discovery_dual import GRNAggDiscoveryDualStrategy
from .strategy_79_critical_period_agg_dual import CriticalPeriodAggDualStrategy
from .strategy_80_hebbian_agg_discovery_dual import HebbianAggDiscoveryDualStrategy
from .strategy_81_synaptic_tagging_agg_dual import SynapticTaggingAggDualStrategy
from .strategy_82_homeostatic_agg_discovery_dual import HomeostaticAggDiscoveryDualStrategy
from .strategy_83_stdp_agg_discovery_dual import STDPAggDiscoveryDualStrategy

# Hybrid Strategies (Part A)
from .strategy_84_tag_homeostatic_dual import TagHomeostaticDualStrategy
from .strategy_85_tag_reinforcement_dual import TagReinforcementDualStrategy
from .strategy_86_homeostatic_reinforcement_dual import HomeostaticReinforcementDualStrategy
from .strategy_87_triple_hybrid_dual import TripleHybridDualStrategy
from .strategy_88_grn_fixed_dual import GRNFixedDualStrategy
from .strategy_89_stdp_tag_dual import STDPTagDualStrategy

# Bio-Inspired Extensions (Part B)
from .strategy_90_neurogenesis_hybrid_dual import NeurogenesisHybridDualStrategy
from .strategy_91_clonal_hybrid_dual import ClonalHybridDualStrategy
from .strategy_92_neuromod_hybrid_dual import NeuromodHybridDualStrategy

# Extended Investigation (Direction A, B, C)
# High Priority
from .strategy_93_clonal_predictive_dual import ClonalPredictiveDualStrategy
from .strategy_97_stdp_domain_aware_dual import STDPDomainAwareDualStrategy
from .strategy_101_aggregation_led_dual import AggregationLedDualStrategy
# Medium Priority
from .strategy_94_clonal_dendritic_dual import ClonalDendriticDualStrategy
from .strategy_99_stdp_no_tag_dual import STDPNoTagDualStrategy
from .strategy_103_cross_domain_v2_dual import CrossDomainV2DualStrategy
# Exploratory
from .strategy_95_clonal_retrograde_dual import ClonalRetrogradeDualStrategy
from .strategy_96_clonal_bcm_dual import ClonalBCMDualStrategy
from .strategy_98_stdp_extended_window_dual import STDPExtendedWindowDualStrategy
from .strategy_100_stdp_adaptive_window_dual import STDPAdaptiveWindowDualStrategy
from .strategy_102_dual_independent_homeostasis import DualIndependentHomeostasisStrategy
from .strategy_104_aggregation_anchor_dual import AggregationAnchorDualStrategy

# Extended Bio-Inspired Dual Strategies
# High Priority
from .strategy_105_keystone_agg_led_dual import KeystoneAggLedDualStrategy
from .strategy_109_grn_cross_v2_dual import GRNCrossV2DualStrategy
from .strategy_113_neurogenesis_agg_led_dual import NeurogenesisAggLedDualStrategy
# Medium Priority
from .strategy_106_keystone_cross_v2_dual import KeystoneCrossV2DualStrategy
from .strategy_111_circadian_clonal_agg_dual import CircadianClonalAggDualStrategy
from .strategy_114_cross_reactive_agg_dual import CrossReactiveAggDualStrategy
# Exploratory
from .strategy_107_keystone_quorum_dual import KeystoneQuorumDualStrategy
from .strategy_108_succession_agg_led_dual import SuccessionAggLedDualStrategy
from .strategy_110_morphogen_agg_led_dual import MorphogenAggLedDualStrategy
from .strategy_112_grn_quorum_minority_dual import GRNQuorumMinorityDualStrategy
from .strategy_115_immune_clonal_cross_v2_dual import ImmuneClonalCrossV2DualStrategy
from .strategy_116_neurogenesis_cross_reactive_dual import NeurogenesisCrossReactiveDualStrategy

# High Priority - New Hybrid Bio-Mechanisms
from .strategy_117_stdp_neurogenesis_survival_dual import STDPNeurogenesisSurvivalDualStrategy
from .strategy_121_morphogen_critical_period_dual import MorphogenCriticalPeriodDualStrategy
from .strategy_125_succession_immune_pioneer_dual import SuccessionImmunePioneerDualStrategy
from .strategy_128_bcm_aggregation_dual import BCMAggregationDualStrategy

__all__ = [
    'PaletteEvolutionStrategy',
    'BaselineStrategy',
    'FitnessGuidedStrategy',
    'AsymmetricStickyStrategy',
    'SinDefaultStrategy',
    'SeparateGenomeStrategy',
    'ExplorationBonusStrategy',
    'NeuromodulatedStrategy',
    'HebbianStrategy',
    'CriticalPeriodStrategy',
    'NeuroHebbianStrategy',
    'CriticalStickyStrategy',
    'CriticalHebbianStrategy',
    'HomeostaticHebbianStrategy',
    'DecayingHebbianStrategy',
    'MetaplasticStrategy',
    'STDPStrategy',
    'CompetitiveHebbianStrategy',
    'NeuralDarwinismStrategy',
    'ConsolidationWindowStrategy',
    'SynapticTaggingStrategy',
    'MultiNeuromodulatoryStrategy',
    'EligibilityTraceStrategy',
    'ComplementaryLearningStrategy',
    'PredictiveCodingStrategy',
    'IntrinsicPlasticityStrategy',
    'EcologicalSuccessionStrategy',
    'QuorumSensingStrategy',
    'DendriticComputationStrategy',
    'ClonalSelectionStrategy',
    'ClonalSelectionMemcellStrategy',
    'PredatorPreyStrategy',
    'MorphogenGradientStrategy',
    # Strategies
    'AntColonyPheromoneStrategy',
    'GeneticRegulatoryNetworkStrategy',
    'CircadianRhythmStrategy',
    # Strategies
    'SynapticFatigueStrategy',
    'BurstRefractoryStrategy',
    'RetrogradeSignalingStrategy',
    'GlialModulationStrategy',
    'AnchorInhibitionStrategy',
    'ProgressivePaletteStrategy',
    # Strategies
    'AdultNeurogenesisStrategy',
    'SynapticHomeostasisHypothesisStrategy',
    'OpioidNeuromodulationStrategy',
    'ImmuneMemoryStrategy',
    'CriticalPeriodRefinedStrategy',
    'EcosystemDynamicsStrategy',
    # Dual Palette Strategies
    'NeuromodulatedDualStrategy',
    'HebbianDualStrategy',
    'CriticalPeriodDualStrategy',
    'CriticalHebbianDualStrategy',
    'HomeostaticHebbianDualStrategy',
    'DecayingHebbianDualStrategy',
    'MetaplasticDualStrategy',
    'STDPDualStrategy',
    'CompetitiveHebbianDualStrategy',
    'NeuralDarwinismDualStrategy',
    'ConsolidationWindowDualStrategy',
    'SynapticTaggingDualStrategy',
    'MultiNeuromodulatoryDualStrategy',
    'EligibilityTraceDualStrategy',
    'ComplementaryLearningDualStrategy',
    'PredictiveCodingDualStrategy',
    'IntrinsicPlasticityDualStrategy',
    'EcologicalSuccessionDualStrategy',
    'DendriticComputationDualStrategy',
    'MorphogenGradientDualStrategy',
    'QuorumSensingDualStrategy',
    'ClonalSelectionDualStrategy',
    'PredatorPreyDualStrategy',
    'ExplorationBonusDualStrategy',
    'AsymmetricStickyDualStrategy',
    'NeuroHebbianDualStrategy',
    # New Dual Strategies
    'BaselineDualStrategy',
    'FitnessGuidedDualStrategy',
    'SinDefaultDualStrategy',
    'AntColonyPheromoneDualStrategy',
    'EcosystemDynamicsDualStrategy',
    'CircadianRhythmDualStrategy',
    'BurstRefractoryDualStrategy',
    'SynapticFatigueDualStrategy',
    'RetrogradeSignalingDualStrategy',
    'GlialModulationDualStrategy',
    'OpioidNeuromodulationDualStrategy',
    'AnchorInhibitionDualStrategy',
    'SynapticHomeostasisDualStrategy',
    'ImmuneMemoryDualStrategy',
    'GeneticRegulatoryNetworkDualStrategy',
    'ProgressivePaletteDualStrategy',
    'AdultNeurogenesisDualStrategy',
    'CriticalPeriodRefinedDualStrategy',
    'SeparateGenomeDualStrategy',
    'CriticalStickyDualStrategy',
    # Hybrid Strategies
    'ProgressiveConsolidationStrategy',
    'EcosystemNeurogenesisStrategy',
    'ClonalImmuneStrategy',
    # Aggregation-Focused Strategies
    'AggHomeostasisDualStrategy',
    'AggGatedRetentionDualStrategy',
    'CrossDomainReinforcementDualStrategy',
    'AggCriticalPeriodDualStrategy',
    'ExtremeValueBiasDualStrategy',
    'AggStabilityLockDualStrategy',
    # Aggregation Discovery Strategies
    'GRNAggDiscoveryDualStrategy',
    'CriticalPeriodAggDualStrategy',
    'HebbianAggDiscoveryDualStrategy',
    'SynapticTaggingAggDualStrategy',
    'HomeostaticAggDiscoveryDualStrategy',
    'STDPAggDiscoveryDualStrategy',
    # Hybrid Strategies (Part A)
    'TagHomeostaticDualStrategy',
    'TagReinforcementDualStrategy',
    'HomeostaticReinforcementDualStrategy',
    'TripleHybridDualStrategy',
    'GRNFixedDualStrategy',
    'STDPTagDualStrategy',
    # Bio-Inspired Extensions (Part B)
    'NeurogenesisHybridDualStrategy',
    'ClonalHybridDualStrategy',
    'NeuromodHybridDualStrategy',
    # Extended Investigation
    'ClonalPredictiveDualStrategy',
    'STDPDomainAwareDualStrategy',
    'AggregationLedDualStrategy',
    'ClonalDendriticDualStrategy',
    'STDPNoTagDualStrategy',
    'CrossDomainV2DualStrategy',
    'ClonalRetrogradeDualStrategy',
    'ClonalBCMDualStrategy',
    'STDPExtendedWindowDualStrategy',
    'STDPAdaptiveWindowDualStrategy',
    'DualIndependentHomeostasisStrategy',
    'AggregationAnchorDualStrategy',
    # Extended Bio-Inspired Dual Strategies
    'KeystoneAggLedDualStrategy',
    'GRNCrossV2DualStrategy',
    'NeurogenesisAggLedDualStrategy',
    # Extended Bio-Inspired Dual Strategies
    'KeystoneCrossV2DualStrategy',
    'CircadianClonalAggDualStrategy',
    'CrossReactiveAggDualStrategy',
    # Extended Bio-Inspired Dual Strategies
    'KeystoneQuorumDualStrategy',
    'SuccessionAggLedDualStrategy',
    'MorphogenAggLedDualStrategy',
    'GRNQuorumMinorityDualStrategy',
    'ImmuneClonalCrossV2DualStrategy',
    'NeurogenesisCrossReactiveDualStrategy',
    # Hybrid Bio-Mechanism Strategies
    'STDPNeurogenesisSurvivalDualStrategy',
    'MorphogenCriticalPeriodDualStrategy',
    'SuccessionImmunePioneerDualStrategy',
    'BCMAggregationDualStrategy',
]

# Quick access dict
STRATEGIES = {
    'baseline': BaselineStrategy,
    'fitness_guided': FitnessGuidedStrategy,
    'asymmetric_sticky': AsymmetricStickyStrategy,
    'sin_default': SinDefaultStrategy,
    'separate_genome': SeparateGenomeStrategy,
    'exploration_bonus': ExplorationBonusStrategy,
    'neuromodulated': NeuromodulatedStrategy,
    'hebbian': HebbianStrategy,
    'critical_period': CriticalPeriodStrategy,
    'neuro_hebbian': NeuroHebbianStrategy,
    'critical_sticky': CriticalStickyStrategy,
    'critical_hebbian': CriticalHebbianStrategy,
    'homeostatic_hebbian': HomeostaticHebbianStrategy,
    'decaying_hebbian': DecayingHebbianStrategy,
    'metaplastic': MetaplasticStrategy,
    'stdp': STDPStrategy,
    'competitive_hebbian': CompetitiveHebbianStrategy,
    'neural_darwinism': NeuralDarwinismStrategy,
    'consolidation_window': ConsolidationWindowStrategy,
    'synaptic_tagging': SynapticTaggingStrategy,
    'multi_neuromodulatory': MultiNeuromodulatoryStrategy,
    'eligibility_trace': EligibilityTraceStrategy,
    'complementary_learning': ComplementaryLearningStrategy,
    'predictive_coding': PredictiveCodingStrategy,
    'intrinsic_plasticity': IntrinsicPlasticityStrategy,
    'ecological_succession': EcologicalSuccessionStrategy,
    'quorum_sensing': QuorumSensingStrategy,
    'dendritic_computation': DendriticComputationStrategy,
    'clonal_selection': ClonalSelectionStrategy,
    'clonal_selection_memcell': ClonalSelectionMemcellStrategy,
    'predator_prey': PredatorPreyStrategy,
    'morphogen_gradient': MorphogenGradientStrategy,
    # Strategies
    'ant_colony_pheromone': AntColonyPheromoneStrategy,
    'genetic_regulatory_network': GeneticRegulatoryNetworkStrategy,
    'circadian_rhythm': CircadianRhythmStrategy,
    # Strategies
    'synaptic_fatigue': SynapticFatigueStrategy,
    'burst_refractory': BurstRefractoryStrategy,
    'retrograde_signaling': RetrogradeSignalingStrategy,
    'glial_modulation': GlialModulationStrategy,
    'anchor_inhibition': AnchorInhibitionStrategy,
    'progressive_palette': ProgressivePaletteStrategy,
    # Strategies
    'adult_neurogenesis': AdultNeurogenesisStrategy,
    'synaptic_homeostasis': SynapticHomeostasisHypothesisStrategy,
    'opioid_neuromodulation': OpioidNeuromodulationStrategy,
    'immune_memory': ImmuneMemoryStrategy,
    'critical_period_refined': CriticalPeriodRefinedStrategy,
    'ecosystem_dynamics': EcosystemDynamicsStrategy,
    # Dual Palette Strategies
    'neuromodulated_dual': NeuromodulatedDualStrategy,
    'hebbian_dual': HebbianDualStrategy,
    'critical_period_dual': CriticalPeriodDualStrategy,
    'critical_hebbian_dual': CriticalHebbianDualStrategy,
    'homeostatic_hebbian_dual': HomeostaticHebbianDualStrategy,
    'decaying_hebbian_dual': DecayingHebbianDualStrategy,
    'metaplastic_dual': MetaplasticDualStrategy,
    'stdp_dual': STDPDualStrategy,
    'competitive_hebbian_dual': CompetitiveHebbianDualStrategy,
    'neural_darwinism_dual': NeuralDarwinismDualStrategy,
    'consolidation_window_dual': ConsolidationWindowDualStrategy,
    'synaptic_tagging_dual': SynapticTaggingDualStrategy,
    'multi_neuromodulatory_dual': MultiNeuromodulatoryDualStrategy,
    'eligibility_trace_dual': EligibilityTraceDualStrategy,
    'complementary_learning_dual': ComplementaryLearningDualStrategy,
    'predictive_coding_dual': PredictiveCodingDualStrategy,
    'intrinsic_plasticity_dual': IntrinsicPlasticityDualStrategy,
    'ecological_succession_dual': EcologicalSuccessionDualStrategy,
    'dendritic_computation_dual': DendriticComputationDualStrategy,
    'morphogen_gradient_dual': MorphogenGradientDualStrategy,
    'quorum_sensing_dual': QuorumSensingDualStrategy,
    'clonal_selection_dual': ClonalSelectionDualStrategy,
    'predator_prey_dual': PredatorPreyDualStrategy,
    'exploration_bonus_dual': ExplorationBonusDualStrategy,
    'asymmetric_sticky_dual': AsymmetricStickyDualStrategy,
    'neuro_hebbian_dual': NeuroHebbianDualStrategy,
    # New Dual Strategies
    'baseline_dual': BaselineDualStrategy,
    'fitness_guided_dual': FitnessGuidedDualStrategy,
    'sin_default_dual': SinDefaultDualStrategy,
    'ant_colony_pheromone_dual': AntColonyPheromoneDualStrategy,
    'ecosystem_dynamics_dual': EcosystemDynamicsDualStrategy,
    'circadian_rhythm_dual': CircadianRhythmDualStrategy,
    'burst_refractory_dual': BurstRefractoryDualStrategy,
    'synaptic_fatigue_dual': SynapticFatigueDualStrategy,
    'retrograde_signaling_dual': RetrogradeSignalingDualStrategy,
    'glial_modulation_dual': GlialModulationDualStrategy,
    'opioid_neuromodulation_dual': OpioidNeuromodulationDualStrategy,
    'anchor_inhibition_dual': AnchorInhibitionDualStrategy,
    'synaptic_homeostasis_dual': SynapticHomeostasisDualStrategy,
    'immune_memory_dual': ImmuneMemoryDualStrategy,
    'genetic_regulatory_network_dual': GeneticRegulatoryNetworkDualStrategy,
    'progressive_palette_dual': ProgressivePaletteDualStrategy,
    'adult_neurogenesis_dual': AdultNeurogenesisDualStrategy,
    'critical_period_refined_dual': CriticalPeriodRefinedDualStrategy,
    'separate_genome_dual': SeparateGenomeDualStrategy,
    'critical_sticky_dual': CriticalStickyDualStrategy,
    # Hybrid Strategies
    'progressive_consolidation': ProgressiveConsolidationStrategy,
    'ecosystem_neurogenesis': EcosystemNeurogenesisStrategy,
    'clonal_immune': ClonalImmuneStrategy,
    # Aggregation-Focused Strategies
    'agg_homeostasis_dual': AggHomeostasisDualStrategy,
    'agg_gated_retention_dual': AggGatedRetentionDualStrategy,
    'cross_domain_reinforcement_dual': CrossDomainReinforcementDualStrategy,
    'agg_critical_period_dual': AggCriticalPeriodDualStrategy,
    'extreme_value_bias_dual': ExtremeValueBiasDualStrategy,
    'agg_stability_lock_dual': AggStabilityLockDualStrategy,
    # Aggregation Discovery Strategies
    'grn_agg_discovery_dual': GRNAggDiscoveryDualStrategy,
    'critical_period_agg_dual': CriticalPeriodAggDualStrategy,
    'hebbian_agg_discovery_dual': HebbianAggDiscoveryDualStrategy,
    'synaptic_tagging_agg_dual': SynapticTaggingAggDualStrategy,
    'homeostatic_agg_discovery_dual': HomeostaticAggDiscoveryDualStrategy,
    'stdp_agg_discovery_dual': STDPAggDiscoveryDualStrategy,
    # Hybrid Strategies (Part A)
    'tag_homeostatic_dual': TagHomeostaticDualStrategy,
    'tag_reinforcement_dual': TagReinforcementDualStrategy,
    'homeostatic_reinforcement_dual': HomeostaticReinforcementDualStrategy,
    'triple_hybrid_dual': TripleHybridDualStrategy,
    'grn_fixed_dual': GRNFixedDualStrategy,
    'stdp_tag_dual': STDPTagDualStrategy,
    # Bio-Inspired Extensions (Part B)
    'neurogenesis_hybrid_dual': NeurogenesisHybridDualStrategy,
    'clonal_hybrid_dual': ClonalHybridDualStrategy,
    'neuromod_hybrid_dual': NeuromodHybridDualStrategy,
    # Extended Investigation
    'clonal_predictive_dual': ClonalPredictiveDualStrategy,
    'stdp_domain_aware_dual': STDPDomainAwareDualStrategy,
    'aggregation_led_dual': AggregationLedDualStrategy,
    'clonal_dendritic_dual': ClonalDendriticDualStrategy,
    'stdp_no_tag_dual': STDPNoTagDualStrategy,
    'cross_domain_v2_dual': CrossDomainV2DualStrategy,
    'clonal_retrograde_dual': ClonalRetrogradeDualStrategy,
    'clonal_bcm_dual': ClonalBCMDualStrategy,
    'stdp_extended_window_dual': STDPExtendedWindowDualStrategy,
    'stdp_adaptive_window_dual': STDPAdaptiveWindowDualStrategy,
    'dual_independent_homeostasis': DualIndependentHomeostasisStrategy,
    'aggregation_anchor_dual': AggregationAnchorDualStrategy,
    # Extended Bio-Inspired Dual Strategies
    'keystone_agg_led_dual': KeystoneAggLedDualStrategy,
    'grn_cross_v2_dual': GRNCrossV2DualStrategy,
    'neurogenesis_agg_led_dual': NeurogenesisAggLedDualStrategy,
    # Extended Bio-Inspired Dual Strategies
    'keystone_cross_v2_dual': KeystoneCrossV2DualStrategy,
    'circadian_clonal_agg_dual': CircadianClonalAggDualStrategy,
    'cross_reactive_agg_dual': CrossReactiveAggDualStrategy,
    # Extended Bio-Inspired Dual Strategies
    'keystone_quorum_dual': KeystoneQuorumDualStrategy,
    'succession_agg_led_dual': SuccessionAggLedDualStrategy,
    'morphogen_agg_led_dual': MorphogenAggLedDualStrategy,
    'grn_quorum_minority_dual': GRNQuorumMinorityDualStrategy,
    'immune_clonal_cross_v2_dual': ImmuneClonalCrossV2DualStrategy,
    'neurogenesis_cross_reactive_dual': NeurogenesisCrossReactiveDualStrategy,
}
