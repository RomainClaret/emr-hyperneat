#!/usr/bin/env python3
"""Classification problems for Dynamic Functions evaluation.

This module provides classification problems beyond XOR for testing
whether CPPN-derived activation functions provide task-appropriate diversity.

Key Problems:
- Two Spirals: Requires both periodic (angular) and non-periodic (radial) processing
- Step Function: Tests sharp boundary detection
- Concentric Circles: Tests radial pattern recognition
- Parity (N-bit): Tests deep logical reasoning

Each problem matches the interface expected by HMR-HyperNEAT:
- input_shape, output_shape: Tuple dimensions
- jitable, use_bias: Boolean flags
- get_data(): Returns list of (input, target) tuples
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple

# Data directory for DES-HyperNEAT benchmark datasets
_DES_DATA_DIR = Path(__file__).resolve().parents[3] / 'external' / 'des-hyperneat-rust' / 'datasets'


class XORProblem:
    """Standard XOR problem (baseline)."""
    input_shape = (3,)  # 2 inputs + bias
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = 0.95  # Standard NEAT benchmark

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[0.0], [1.0], [1.0], [0.0]]

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class TwoSpiralsProblem:
    """Two interleaved spirals classification.

    Primary test for Dynamic Functions: requires both:
    - Periodic processing (angular component) → sin activation
    - Non-periodic processing (radial component) → tanh/relu activation

    The CPPN should assign different activations based on spatial position.

    Args:
        n_points_per_spiral: Number of points per spiral class (default 97)
        noise: Gaussian noise standard deviation (default 0.0)
        use_bias: Whether to include bias term in inputs
    """
    jitable = True
    fitness_threshold = 0.85  # Very difficult problem, 194 points

    def __init__(self, n_points_per_spiral: int = 97, noise: float = 0.0, use_bias: bool = True):
        self.n_points = n_points_per_spiral
        self.noise = noise
        self.use_bias = use_bias

        # Generate spirals
        inputs, targets = self._generate_spirals()

        self.inputs = inputs
        self.targets = targets

        # Set shapes based on bias setting
        self.input_shape = (3,) if use_bias else (2,)
        self.output_shape = (1,)

    def _generate_spirals(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate two interleaved spirals.

        Spiral 0: Points along one arm
        Spiral 1: Points along second arm (180° offset)

        Parametric equations:
            r(t) = a + b*t  (Archimedean spiral)
            x = r(t) * cos(t)
            y = r(t) * sin(t)
        """
        inputs = []
        targets = []

        np.random.seed(42)  # Reproducibility

        for spiral_class in [0, 1]:
            for i in range(self.n_points):
                # Angle from 0 to 3*pi (1.5 rotations)
                angle = i * np.pi / 16.0

                # Radius grows linearly with angle
                radius = 6.5 * (104 - i) / 104.0

                # Phase offset for second spiral
                offset = np.pi * spiral_class

                # Coordinates
                x = radius * np.sin(angle + offset)
                y = radius * np.cos(angle + offset)

                # Add noise
                if self.noise > 0:
                    x += np.random.normal(0, self.noise)
                    y += np.random.normal(0, self.noise)

                # Normalize to [-1, 1] range
                x_norm = x / 6.5
                y_norm = y / 6.5

                # Add to dataset
                if self.use_bias:
                    inputs.append([x_norm, y_norm, 1.0])
                else:
                    inputs.append([x_norm, y_norm])
                targets.append([float(spiral_class)])

        return inputs, targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class ContinuousTwoSpiralsProblem:
    """Two spirals with continuous output reformulation.

    This is a reformulation of TwoSpiralsProblem to test the hypothesis:
    "CPPNs can learn spirals if output is continuous instead of discrete"

    Instead of binary classification (0 or 1), this outputs a continuous
    "spiral phase" value in [0, 1] representing the angular position relative
    to the spiral pattern:
    - 0.0 = exactly on spiral 0 trajectory
    - 1.0 = exactly on spiral 1 trajectory
    - Intermediate values = between spirals

    The continuous target is computed as:
        phase_diff = actual_angle - expected_spiral0_angle
        target = 0.5 * (1 - cos(phase_diff))

    This gives smooth [0, 1] values that CPPNs can naturally approximate.

    Why this might work better: The binary classification requires sharp
    decision boundaries between interleaved spirals. Continuous phase
    outputs match CPPN's ability to produce smooth functions.

    NOTE: This reformulation changes what the network learns:
    - Binary: "which spiral class?"
    - Continuous: "where in the spiral pattern?"
    Both require understanding the spiral structure, but continuous
    is more compatible with CPPN's smooth function approximation.
    """

    jitable = True
    fitness_threshold = 0.90  # Should be easier with continuous output
    recommended_fitness_mode = 'mse'  # Regression (continuous targets)

    def __init__(self, n_points_per_spiral: int = 97, noise: float = 0.0, use_bias: bool = True):
        self.n_points = n_points_per_spiral
        self.noise = noise
        self.use_bias = use_bias

        # Generate spirals with continuous targets
        inputs, targets = self._generate_spirals()

        self.inputs = inputs
        self.targets = targets

        self.input_shape = (3,) if use_bias else (2,)
        self.output_shape = (1,)

    def _generate_spirals(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate spirals with continuous phase-based targets.

        For each point, compute its "spiral phase" - a continuous [0, 1] value
        that is 0.0 for spiral 0 positions and 1.0 for spiral 1 positions.
        """
        inputs = []
        targets = []

        np.random.seed(42)

        for spiral_class in [0, 1]:
            for i in range(self.n_points):
                # Angle from 0 to ~6*pi (3 rotations)
                angle = i * np.pi / 16.0

                # Radius decreases with i
                radius = 6.5 * (104 - i) / 104.0

                # Phase offset for second spiral
                offset = np.pi * spiral_class

                # Coordinates (note: sin for x, cos for y)
                x = radius * np.sin(angle + offset)
                y = radius * np.cos(angle + offset)

                # Add noise
                if self.noise > 0:
                    x += np.random.normal(0, self.noise)
                    y += np.random.normal(0, self.noise)

                # Normalize to [-1, 1]
                x_norm = x / 6.5
                y_norm = y / 6.5

                # Compute continuous target based on actual position
                # This represents "where in the spiral phase" the point is
                continuous_target = self._compute_spiral_phase(x_norm, y_norm)

                if self.use_bias:
                    inputs.append([x_norm, y_norm, 1.0])
                else:
                    inputs.append([x_norm, y_norm])

                targets.append([continuous_target])

        return inputs, targets

    def _compute_spiral_phase(self, x_norm: float, y_norm: float) -> float:
        """Compute continuous spiral phase for a point.

        Returns a value in [0, 1] where:
        - 0.0 = exactly on spiral 0
        - 1.0 = exactly on spiral 1
        - Intermediate values smoothly interpolate

        The phase is computed from the angular difference between
        the actual angle and the expected angle for spiral 0.
        """
        # Recover unnormalized coordinates
        x = x_norm * 6.5
        y = y_norm * 6.5

        # Compute radius
        r = np.sqrt(x**2 + y**2)

        if r < 0.01:  # Near origin, undefined
            return 0.5

        # Compute actual angle (matching the parameterization: sin for x, cos for y)
        # atan2(y, x) gives angle where y = r*sin(θ), x = r*cos(θ)
        # But our spiral uses x = r*sin(θ), y = r*cos(θ)
        # So actual_angle = atan2(x, y)
        actual_angle = np.arctan2(x, y)

        # From radius, estimate the i index: radius = 6.5 * (104 - i) / 104
        # => i = 104 - (r * 104 / 6.5)
        estimated_i = max(0, min(96, 104 - (r * 104 / 6.5)))

        # Expected angle for spiral 0 at this radius
        expected_angle_spiral0 = estimated_i * np.pi / 16.0

        # Angular difference (wrapped to [-π, π])
        phase_diff = actual_angle - expected_angle_spiral0

        # Wrap to [-π, π]
        while phase_diff > np.pi:
            phase_diff -= 2 * np.pi
        while phase_diff < -np.pi:
            phase_diff += 2 * np.pi

        # Convert to [0, 1] using cosine mapping:
        # phase_diff = 0 → spiral 0 → target = 0
        # phase_diff = π → spiral 1 → target = 1
        # Smooth transition between
        continuous_target = 0.5 * (1 - np.cos(phase_diff))

        return float(continuous_target)

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))

    def get_problem_info(self) -> dict:
        """Get information about this problem configuration."""
        return {
            'name': 'Continuous Two Spirals (Phase)',
            'n_points_per_spiral': self.n_points,
            'n_samples': len(self.inputs),
            'output_type': 'continuous_phase',
            'hypothesis': 'CPPNs can learn spirals with continuous phase output',
            'fitness_threshold': self.fitness_threshold,
        }


class PolarTwoSpiralsProblem:
    """Two spirals with POLAR coordinate inputs.

    This problem reformulates TwoSpiralsProblem by:
    1. Converting input coordinates from Cartesian (x, y) to Polar (r, θ)
    2. Using the polar coordinates as the actual input features

    In polar coordinates, the spiral decision boundary is MUCH simpler:
    - Cartesian: complex interleaved curves
    - Polar: nearly linear (r correlates with θ, with π phase difference between spirals)

    Input: [r, θ_norm, bias] where:
    - r = sqrt(x² + y²) normalized to [0, 1]
    - θ_norm = atan2(x, y) / π normalized to [-1, 1]
    - bias = 1.0

    Output: [class] (0 or 1)

    The hypothesis is that with polar input features, the CPPN can learn the
    simple relationship: "spiral class depends on (θ mod π) vs r correlation"
    """

    jitable = True
    fitness_threshold = 0.90  # Should be easier with polar inputs

    def __init__(self, n_points_per_spiral: int = 97, noise: float = 0.0, use_bias: bool = True):
        self.n_points = n_points_per_spiral
        self.noise = noise
        self.use_bias = use_bias

        # Generate spirals with POLAR inputs
        inputs, targets = self._generate_spirals()

        self.inputs = inputs
        self.targets = targets

        self.input_shape = (3,) if use_bias else (2,)
        self.output_shape = (1,)

    def _generate_spirals(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate spirals with POLAR coordinate inputs."""
        inputs = []
        targets = []

        np.random.seed(42)

        for spiral_class in [0, 1]:
            for i in range(self.n_points):
                # Angle from 0 to ~6*pi
                angle = i * np.pi / 16.0

                # Radius decreases with i
                radius = 6.5 * (104 - i) / 104.0

                # Phase offset for second spiral
                offset = np.pi * spiral_class

                # Cartesian coordinates (for noise calculation)
                x = radius * np.sin(angle + offset)
                y = radius * np.cos(angle + offset)

                # Add noise
                if self.noise > 0:
                    x += np.random.normal(0, self.noise)
                    y += np.random.normal(0, self.noise)

                # Convert to POLAR coordinates for input
                r = np.sqrt(x**2 + y**2) / 6.5  # Normalize r to [0, 1]
                theta = np.arctan2(x, y)  # θ in [-π, π]
                theta_norm = theta / np.pi  # Normalize to [-1, 1]

                if self.use_bias:
                    inputs.append([r, theta_norm, 1.0])
                else:
                    inputs.append([r, theta_norm])

                targets.append([float(spiral_class)])

        return inputs, targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))

    def get_problem_info(self) -> dict:
        """Get information about this problem configuration."""
        return {
            'name': 'Polar Two Spirals',
            'n_points_per_spiral': self.n_points,
            'n_samples': len(self.inputs),
            'input_type': 'polar (r, θ)',
            'hypothesis': 'Polar inputs make spiral decision boundary simpler',
            'fitness_threshold': self.fitness_threshold,
        }


class PolarAugmentedTwoSpiralsProblem:
    """Two spirals with POLAR + TRIGONOMETRIC inputs.

    Combines the best of both approaches:
    - PolarTwoSpiralsProblem: r, θ_norm (polar coordinates)
    - AugmentedTwoSpiralsProblem: sin(θ), cos(θ) (trigonometric features)

    Input: [r, θ_norm, sin(θ), cos(θ), bias] = 5 features

    Where:
    - r: sqrt(x² + y²) / 6.5, normalized to [0, 1]
    - θ_norm: atan2(x, y) / π, normalized to [-1, 1]
    - sin(θ), cos(θ): periodic features for angular discrimination

    The hypothesis is that combining polar base coordinates with explicit
    trigonometric features provides both:
    1. Natural polar representation of the spiral geometry
    2. Explicit periodic features for handling angular wrapping

    Previous results:
    - Polar alone: 0.767 (+1.3%)
    - Augmented alone: 0.757 (+0.3%)
    """

    jitable = True
    fitness_threshold = 0.90

    def __init__(self, n_points_per_spiral: int = 97, noise: float = 0.0, use_bias: bool = True):
        self.n_points = n_points_per_spiral
        self.noise = noise
        self.use_bias = use_bias

        inputs, targets = self._generate_spirals()

        self.inputs = inputs
        self.targets = targets

        self.input_shape = (5,) if use_bias else (4,)
        self.output_shape = (1,)

    def _generate_spirals(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate spirals with polar + trigonometric inputs."""
        inputs = []
        targets = []

        np.random.seed(42)

        for spiral_class in [0, 1]:
            for i in range(self.n_points):
                # Angle from 0 to ~6*pi
                angle = i * np.pi / 16.0

                # Radius decreases with i
                radius = 6.5 * (104 - i) / 104.0

                # Phase offset for second spiral
                offset = np.pi * spiral_class

                # Cartesian coordinates
                x = radius * np.sin(angle + offset)
                y = radius * np.cos(angle + offset)

                # Add noise
                if self.noise > 0:
                    x += np.random.normal(0, self.noise)
                    y += np.random.normal(0, self.noise)

                # POLAR features
                r = np.sqrt(x**2 + y**2) / 6.5  # Normalize to [0, 1]
                theta = np.arctan2(x, y)  # θ in [-π, π]
                theta_norm = theta / np.pi  # Normalize to [-1, 1]

                # TRIGONOMETRIC features
                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)

                # Combined features: [r, θ_norm, sin(θ), cos(θ), bias]
                if self.use_bias:
                    inputs.append([r, theta_norm, sin_theta, cos_theta, 1.0])
                else:
                    inputs.append([r, theta_norm, sin_theta, cos_theta])

                targets.append([float(spiral_class)])

        return inputs, targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))

    def get_problem_info(self) -> dict:
        """Get information about this problem configuration."""
        return {
            'name': 'Polar + Augmented Two Spirals',
            'n_points_per_spiral': self.n_points,
            'n_samples': len(self.inputs),
            'input_type': 'polar + trig (r, θ, sin, cos)',
            'hypothesis': 'Combining polar and trig features provides additive benefit',
            'fitness_threshold': self.fitness_threshold,
        }


class StepFunctionProblem:
    """Step function classification (sharp boundary detection).

    Tests: Can the network learn sharp decision boundaries?

    Decision boundary: y = 0
    - Points above y=0 → class 1
    - Points below y=0 → class 0

    This is trivial for linear classifiers but tests whether
    dynamic activations can learn sharp vs smooth boundaries.

    Args:
        n_points: Total number of points (default 100)
        use_bias: Whether to include bias term
    """
    jitable = True
    fitness_threshold = 0.95  # Linearly separable, should be easy

    def __init__(self, n_points: int = 100, use_bias: bool = True):
        self.n_points = n_points
        self.use_bias = use_bias

        inputs, targets = self._generate_data()

        self.inputs = inputs
        self.targets = targets

        self.input_shape = (3,) if use_bias else (2,)
        self.output_shape = (1,)

    def _generate_data(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate points above/below y=0 boundary."""
        inputs = []
        targets = []

        np.random.seed(42)

        # Generate uniform points in [-1, 1]²
        for i in range(self.n_points):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)

            # Class based on y position
            class_label = 1.0 if y > 0 else 0.0

            if self.use_bias:
                inputs.append([x, y, 1.0])
            else:
                inputs.append([x, y])
            targets.append([class_label])

        return inputs, targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class ConcentricCirclesProblem:
    """Concentric circles classification (radial pattern).

    Tests: Can the network recognize radial patterns?

    Decision boundary: r = 0.5 (circle)
    - Points inside circle (r < 0.5) → class 0
    - Points outside circle (r > 0.5) → class 1

    Requires: Computing r = sqrt(x² + y²) and comparing to threshold

    Args:
        n_points: Total number of points (default 100)
        boundary_radius: Radius of decision boundary (default 0.5)
        use_bias: Whether to include bias term
    """
    jitable = True
    fitness_threshold = 0.90  # Nonlinear but simpler than spirals

    def __init__(self, n_points: int = 100, boundary_radius: float = 0.5, use_bias: bool = True):
        self.n_points = n_points
        self.boundary_radius = boundary_radius
        self.use_bias = use_bias

        inputs, targets = self._generate_data()

        self.inputs = inputs
        self.targets = targets

        self.input_shape = (3,) if use_bias else (2,)
        self.output_shape = (1,)

    def _generate_data(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate points inside/outside circular boundary."""
        inputs = []
        targets = []

        np.random.seed(42)

        # Generate points with 50/50 class distribution
        n_per_class = self.n_points // 2

        # Inner circle (class 0)
        for _ in range(n_per_class):
            # Random angle
            angle = np.random.uniform(0, 2 * np.pi)
            # Random radius (uniform in area, so sqrt for radius)
            r = np.sqrt(np.random.uniform(0, self.boundary_radius**2 * 0.9))  # 90% to avoid boundary

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            if self.use_bias:
                inputs.append([x, y, 1.0])
            else:
                inputs.append([x, y])
            targets.append([0.0])

        # Outer ring (class 1)
        for _ in range(n_per_class):
            angle = np.random.uniform(0, 2 * np.pi)
            # Radius between boundary and 1.0
            r_min = self.boundary_radius * 1.1  # 110% to avoid boundary
            r_max = 1.0
            r = np.sqrt(np.random.uniform(r_min**2, r_max**2))

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            if self.use_bias:
                inputs.append([x, y, 1.0])
            else:
                inputs.append([x, y])
            targets.append([1.0])

        return inputs, targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class ANDProblem:
    """Standard AND gate problem (for multi-task neuromodulation)."""
    input_shape = (3,)  # 2 inputs + bias
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = 0.95

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[0.0], [0.0], [0.0], [1.0]]

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class ORProblem:
    """Standard OR gate problem (for multi-task neuromodulation)."""
    input_shape = (3,)  # 2 inputs + bias
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = 0.95

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[0.0], [1.0], [1.0], [1.0]]

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class NANDProblem:
    """Standard NAND gate problem (for multi-task neuromodulation)."""
    input_shape = (3,)  # 2 inputs + bias
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = 0.95

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[1.0], [1.0], [1.0], [0.0]]

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class NORProblem:
    """Standard NOR gate problem (for multi-task composition)."""
    input_shape = (3,)  # 2 inputs + bias
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = 0.95

    def __init__(self):
        self.inputs = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        self.targets = [[1.0], [0.0], [0.0], [0.0]]

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class ParityProblem:
    """N-bit Parity problem (XOR generalization).

    Tests: Deep logical reasoning with increasing difficulty.

    Output 1 if odd number of inputs are 1, else 0.

    2-bit parity = XOR
    3-bit parity = XOR(a, XOR(b, c))
    etc.

    Args:
        n_bits: Number of input bits (2, 3, 4, 5, or 6)
        use_bias: Whether to include bias term
    """
    jitable = True

    def __init__(self, n_bits: int = 3, use_bias: bool = True):
        # Set fitness threshold based on difficulty
        if n_bits <= 2:
            self.fitness_threshold = 0.95
        elif n_bits == 3:
            self.fitness_threshold = 0.90
        elif n_bits == 4:
            self.fitness_threshold = 0.85
        else:
            self.fitness_threshold = 0.80

        if n_bits < 2 or n_bits > 10:
            raise ValueError("n_bits must be between 2 and 10")

        self.n_bits = n_bits
        self.use_bias = use_bias

        inputs, targets = self._generate_data()

        self.inputs = inputs
        self.targets = targets

        self.input_shape = (n_bits + 1,) if use_bias else (n_bits,)
        self.output_shape = (1,)

    def _generate_data(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate all 2^n_bits combinations."""
        inputs = []
        targets = []

        for i in range(2 ** self.n_bits):
            # Convert to binary
            bits = [(i >> j) & 1 for j in range(self.n_bits)]

            # Parity = XOR of all bits
            parity = sum(bits) % 2

            if self.use_bias:
                inputs.append([float(b) for b in bits] + [1.0])
            else:
                inputs.append([float(b) for b in bits])
            targets.append([float(parity)])

        return inputs, targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class MultiTaskProblem:
    """Combines multiple problems for TRUE multi-task neuromodulation.

    This enables evolution of a SINGLE network that solves MULTIPLE tasks
    by changing ONLY the external task context signal.

    Key Features:
    - All tasks share same input/output shape (required for shared substrate)
    - Task context is EXTERNAL (one-hot or binary encoding of task ID)
    - Evaluation can be per-task or aggregated across all tasks

    Usage:
        problem = MultiTaskProblem(['xor', 'and', 'or'])
        # Evaluate on specific task with external context
        fitness = evaluate_with_task_context(network, task_id=0, problem=problem)
        # Or evaluate on all tasks
        fitnesses = problem.evaluate_all_tasks(network, eval_func)
    """
    jitable = True
    use_bias = True
    fitness_threshold = 0.90  # Harder than single task

    def __init__(self, problem_names: List[str]):
        """Initialize multi-task problem.

        Args:
            problem_names: List of problem names from PROBLEMS registry
        """
        self.problem_names = problem_names
        self.tasks = [get_problem(name) for name in problem_names]
        self.num_tasks = len(self.tasks)

        # Verify all tasks have compatible shapes
        ref_input_shape = self.tasks[0].input_shape
        ref_output_shape = self.tasks[0].output_shape
        for i, task in enumerate(self.tasks[1:], 1):
            if task.input_shape != ref_input_shape:
                raise ValueError(
                    f"Task {problem_names[i]} has input_shape {task.input_shape}, "
                    f"expected {ref_input_shape}"
                )
            if task.output_shape != ref_output_shape:
                raise ValueError(
                    f"Task {problem_names[i]} has output_shape {task.output_shape}, "
                    f"expected {ref_output_shape}"
                )

        self.input_shape = ref_input_shape
        self.output_shape = ref_output_shape

        # Pre-compute task data
        self._task_data = [task.get_data() for task in self.tasks]
        self._task_inputs = [
            [d[0] for d in data] for data in self._task_data
        ]
        self._task_targets = [
            [d[1] for d in data] for data in self._task_data
        ]

    def get_task_data(self, task_id: int) -> Tuple[List[List[float]], List[List[float]]]:
        """Get inputs and targets for a specific task.

        Args:
            task_id: Task index (0 to num_tasks-1)

        Returns:
            (inputs, targets) tuple
        """
        return self._task_inputs[task_id], self._task_targets[task_id]

    def get_task_context(self, task_id: int, method: str = 'one_hot') -> List[float]:
        """Generate external task context vector.

        This is the KEY to true neuromodulation - context is EXTERNAL,
        not derived from inputs.

        Args:
            task_id: Task index
            method: 'one_hot' or 'binary' encoding

        Returns:
            Task context vector
        """
        if method == 'one_hot':
            context = [0.0] * self.num_tasks
            context[task_id] = 1.0
            return context
        elif method == 'binary':
            # Binary encoding (more compact for many tasks)
            n_bits = max(1, int(np.ceil(np.log2(self.num_tasks + 1))))
            return [float((task_id >> i) & 1) for i in range(n_bits)]
        else:
            raise ValueError(f"Unknown context method: {method}")

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        """Get combined data from all tasks (for compatibility)."""
        all_data = []
        for task_data in self._task_data:
            all_data.extend(task_data)
        return all_data

    @property
    def context_size(self) -> int:
        """Size of task context vector (one-hot by default)."""
        return self.num_tasks


# =============================================================================
# Augmented Coordinate Problems
# =============================================================================
# These variants add pre-computed polar coordinates (r, θ, sin(θ), cos(θ))
# to help networks that struggle with the raw (x, y) representation.


class AugmentedConcentricCirclesProblem:
    """Concentric circles with augmented polar coordinates.

    Adds pre-computed radial features to help networks solve the problem:
    - r = sqrt(x² + y²) - the answer is already computed!
    - θ = atan2(y, x) - angle (normalized to [0, 1])
    - sin(θ), cos(θ) - trigonometric features

    Input: [x, y, r, θ, sin(θ), cos(θ), bias]
    Output: [class] (0 = inside, 1 = outside)

    This tests whether providing the radial feature directly helps.
    If r is provided, the network just needs to learn a threshold comparison.
    """
    jitable = True
    fitness_threshold = 0.95  # Should be much easier with r provided

    def __init__(self, n_points: int = 100, boundary_radius: float = 0.5, use_bias: bool = True):
        self.n_points = n_points
        self.boundary_radius = boundary_radius
        self.use_bias = use_bias

        inputs, targets = self._generate_data()

        self.inputs = inputs
        self.targets = targets

        # 6 features + optional bias: [x, y, r, θ, sin(θ), cos(θ), bias]
        self.input_shape = (7,) if use_bias else (6,)
        self.output_shape = (1,)

    def _generate_data(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate points with augmented polar coordinates."""
        inputs = []
        targets = []

        np.random.seed(42)

        n_per_class = self.n_points // 2

        # Inner circle (class 0)
        for _ in range(n_per_class):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(0, self.boundary_radius**2 * 0.9))

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Normalize θ to [0, 1]
            theta_norm = angle / (2 * np.pi)

            features = [x, y, r, theta_norm, np.sin(angle), np.cos(angle)]
            if self.use_bias:
                features.append(1.0)

            inputs.append(features)
            targets.append([0.0])

        # Outer ring (class 1)
        for _ in range(n_per_class):
            angle = np.random.uniform(0, 2 * np.pi)
            r_min = self.boundary_radius * 1.1
            r_max = 1.0
            r = np.sqrt(np.random.uniform(r_min**2, r_max**2))

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            theta_norm = angle / (2 * np.pi)

            features = [x, y, r, theta_norm, np.sin(angle), np.cos(angle)]
            if self.use_bias:
                features.append(1.0)

            inputs.append(features)
            targets.append([1.0])

        return inputs, targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class RadiusOnlyCirclesProblem:
    """Concentric circles with radius-only input.

    The simplest possible input for circles: just [r, bias]
    Since the decision boundary is purely radial (r = 0.5),
    the network only needs to learn a threshold comparison.

    Input: [r, bias] where r = sqrt(x² + y²)
    Output: [class] (0 = inside, 1 = outside)

    This tests the hypothesis that minimal features help CPPN
    (similar to how polar inputs helped spirals).
    """
    jitable = True
    fitness_threshold = 0.95  # Should be trivial with r directly provided

    def __init__(self, n_points: int = 100, boundary_radius: float = 0.5, use_bias: bool = True):
        self.n_points = n_points
        self.boundary_radius = boundary_radius
        self.use_bias = use_bias

        inputs, targets = self._generate_data()

        self.inputs = inputs
        self.targets = targets

        self.input_shape = (2,) if use_bias else (1,)
        self.output_shape = (1,)

    def _generate_data(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate points with radius-only input."""
        inputs = []
        targets = []

        np.random.seed(42)

        n_per_class = self.n_points // 2

        # Inner circle (class 0)
        for _ in range(n_per_class):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(0, self.boundary_radius**2 * 0.9))

            # Just radius, normalized to [0, 1]
            r_norm = r  # Already in [0, ~0.45] range
            features = [r_norm]
            if self.use_bias:
                features.append(1.0)

            inputs.append(features)
            targets.append([0.0])

        # Outer ring (class 1)
        for _ in range(n_per_class):
            angle = np.random.uniform(0, 2 * np.pi)
            r_min = self.boundary_radius * 1.1
            r_max = 1.0
            r = np.sqrt(np.random.uniform(r_min**2, r_max**2))

            r_norm = r  # In [0.55, 1.0] range
            features = [r_norm]
            if self.use_bias:
                features.append(1.0)

            inputs.append(features)
            targets.append([1.0])

        return inputs, targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class CountsRetinaProblem:
    """Retina problem with pre-computed counts (minimal features).

    Tests the Minimal Feature Principle on an already-solved problem.
    Instead of 8 raw pixels + bias = 9 features, provides:
    Input: [left_count_norm, right_count_norm, bias] = 3 features

    The standard retina requires the network to:
    1. Separate left/right pixels (spatial awareness)
    2. Count activations per side
    3. Compare counts

    This variant pre-computes steps 1-2, leaving only comparison.
    """
    jitable = True
    fitness_threshold = 0.95
    input_shape = (3,)  # left_count_norm, right_count_norm, bias
    output_shape = (2,)  # [left_has_more, right_has_more]

    def __init__(self, use_bias: bool = True):
        self.use_bias = use_bias
        if not use_bias:
            self.input_shape = (2,)

        self._generate_data()

    def _generate_data(self):
        """Generate all 256 patterns with pre-computed counts."""
        self.inputs = []
        self.targets = []

        for i in range(256):
            bits = [(i >> j) & 1 for j in range(8)]

            left_sum = sum(bits[:4])
            right_sum = sum(bits[4:])

            # Normalize counts to [0, 1]
            left_norm = left_sum / 4.0
            right_norm = right_sum / 4.0

            if left_sum > right_sum:
                target = [1.0, 0.0]
            elif right_sum > left_sum:
                target = [0.0, 1.0]
            else:
                target = [0.5, 0.5]

            features = [left_norm, right_norm]
            if self.use_bias:
                features.append(1.0)

            self.inputs.append(features)
            self.targets.append(target)

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class CompoundBooleanProblem:
    """Compound Boolean task: op1(a,b) OP op2(c,d) [OP op3(e,f)].

    Generates truth table for compound tasks:
    - 2-op: 4 inputs, 16 rows (default)
    - 3-op: 6 inputs, 64 rows (when sub_op3 is provided)

    Args:
        sub_op1: First sub-operation ('xor', 'and', 'or', 'nand')
        sub_op2: Second sub-operation ('xor', 'and', 'or', 'nand')
        compose_op: Composition operation ('and', 'or', 'xor')
        sub_op3: Optional third sub-operation for 3-grid composition
        use_bias: Whether to include bias term
    """
    jitable = True
    fitness_threshold = 0.95
    output_shape = (1,)

    _OPS = {
        'xor': lambda a, b: a ^ b,
        'and': lambda a, b: a & b,
        'or': lambda a, b: a | b,
        'nand': lambda a, b: int(not (a & b)),
        'nor': lambda a, b: int(not (a | b)),
    }

    def __init__(self, sub_op1: str = 'xor', sub_op2: str = 'and',
                 compose_op: str = 'and', sub_op3: str = None,
                 use_bias: bool = True):
        self.sub_op1 = sub_op1
        self.sub_op2 = sub_op2
        self.compose_op = compose_op
        self.sub_op3 = sub_op3
        self.use_bias = use_bias
        n_raw = 6 if sub_op3 else 4
        self.input_shape = (n_raw + 1,) if use_bias else (n_raw,)

        op1 = self._OPS[sub_op1]
        op2 = self._OPS[sub_op2]
        compose = self._OPS[compose_op]
        op3 = self._OPS[sub_op3] if sub_op3 else None

        self.inputs = []
        self.targets = []
        if sub_op3:
            # 3-op: 6 inputs, 64 rows, left-associative composition
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        for d in (0, 1):
                            for e in (0, 1):
                                for f in (0, 1):
                                    r1 = op1(a, b)
                                    r2 = op2(c, d)
                                    r3 = op3(e, f)
                                    result = compose(compose(r1, r2), r3)
                                    inp = [float(a), float(b), float(c),
                                           float(d), float(e), float(f)]
                                    if use_bias:
                                        inp.append(1.0)
                                    self.inputs.append(inp)
                                    self.targets.append([float(result)])
        else:
            # 2-op: 4 inputs, 16 rows
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        for d in (0, 1):
                            r1 = op1(a, b)
                            r2 = op2(c, d)
                            result = compose(r1, r2)
                            inp = [float(a), float(b), float(c), float(d)]
                            if use_bias:
                                inp.append(1.0)
                            self.inputs.append(inp)
                            self.targets.append([float(result)])

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))

    def get_sub_problems(self) -> Tuple['XORProblem', 'ANDProblem']:
        """Return the two sub-problem instances (for grid evolution)."""
        return get_problem(self.sub_op1), get_problem(self.sub_op2)


class AugmentedTwoSpiralsProblem:
    """Two spirals with augmented polar coordinates.

    Adds pre-computed polar features to help with angular/radial decomposition:
    - r = sqrt(x² + y²) - radial distance
    - θ = atan2(y, x) - angle (normalized to [0, 1])
    - sin(θ), cos(θ) - trigonometric features for periodic processing

    Input: [x, y, r, θ, sin(θ), cos(θ), bias]
    Output: [class] (0 or 1)

    The spiral problem requires separating points that are interleaved angularly
    but at the same radius. With sin(θ) and cos(θ) provided, the angular
    separation becomes much easier.
    """
    jitable = True
    fitness_threshold = 0.90  # Should be easier with polar coords

    def __init__(self, n_points_per_spiral: int = 97, noise: float = 0.0, use_bias: bool = True):
        self.n_points = n_points_per_spiral
        self.noise = noise
        self.use_bias = use_bias

        inputs, targets = self._generate_spirals()

        self.inputs = inputs
        self.targets = targets

        # 6 features + optional bias
        self.input_shape = (7,) if use_bias else (6,)
        self.output_shape = (1,)

    def _generate_spirals(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate spirals with augmented polar coordinates."""
        inputs = []
        targets = []

        np.random.seed(42)

        for spiral_class in [0, 1]:
            for i in range(self.n_points):
                # Angle from 0 to ~3*pi (1.5 rotations)
                angle = i * np.pi / 16.0

                # Radius grows linearly with angle
                radius = 6.5 * (104 - i) / 104.0

                # Phase offset for second spiral
                offset = np.pi * spiral_class

                # Cartesian coordinates
                x = radius * np.sin(angle + offset)
                y = radius * np.cos(angle + offset)

                # Add noise
                if self.noise > 0:
                    x += np.random.normal(0, self.noise)
                    y += np.random.normal(0, self.noise)

                # Normalize to [-1, 1] range
                x_norm = x / 6.5
                y_norm = y / 6.5

                # Compute polar coordinates from normalized Cartesian
                r = np.sqrt(x_norm**2 + y_norm**2)
                theta = np.arctan2(y_norm, x_norm)
                theta_norm = (theta + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

                features = [x_norm, y_norm, r, theta_norm, np.sin(theta), np.cos(theta)]
                if self.use_bias:
                    features.append(1.0)

                inputs.append(features)
                targets.append([float(spiral_class)])

        return inputs, targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class IrisProblem:
    """Iris classification (DES-HyperNEAT benchmark, Tenstad & Glette 2021).

    4 raw (unnormalized) features + bias -> 3 one-hot class outputs.
    150 samples from Fisher's Iris dataset.
    Class mapping matches DES paper: setosa->[0,0,1], versicolor->[0,1,0], virginica->[1,0,0].
    """
    input_shape = (5,)  # 4 features + 1 bias
    output_shape = (3,)
    jitable = True
    use_bias = True
    fitness_threshold = 0.85

    CLASS_MAP = {
        'Iris-setosa': [0.0, 0.0, 1.0],
        'Iris-versicolor': [0.0, 1.0, 0.0],
        'Iris-virginica': [1.0, 0.0, 0.0],
    }

    def __init__(self):
        self.inputs = []
        self.targets = []
        data_path = _DES_DATA_DIR / 'iris.data'
        if not data_path.exists():
            raise FileNotFoundError(
                f"Iris data not found at {data_path}. "
                "Ensure the des-hyperneat-rust submodule is initialized: "
                "git submodule update --init"
            )
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                features = [float(x) for x in parts[:4]]
                class_name = parts[4].strip()
                self.inputs.append(features + [1.0])
                self.targets.append(self.CLASS_MAP[class_name])

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class WineProblem:
    """Wine classification (DES-HyperNEAT benchmark, Tenstad & Glette 2021).

    13 min-max normalized features + bias -> 3 one-hot class outputs.
    178 samples from UCI Wine dataset.
    Class mapping: 1->[0,0,1], 2->[0,1,0], 3->[1,0,0].

    Features are normalized to [0, 1] per feature because raw values span
    [0.1, 1680], proline alone reaches 1680, causing activation saturation
    and NaN propagation in indirect encoding substrates.
    """
    input_shape = (14,)  # 13 features + 1 bias
    output_shape = (3,)
    jitable = True
    use_bias = True
    fitness_threshold = 0.80

    CLASS_MAP = {
        1: [0.0, 0.0, 1.0],
        2: [0.0, 1.0, 0.0],
        3: [1.0, 0.0, 0.0],
    }

    def __init__(self):
        self.inputs = []
        self.targets = []
        data_path = _DES_DATA_DIR / 'wine.data'
        if not data_path.exists():
            raise FileNotFoundError(
                f"Wine data not found at {data_path}. "
                "Ensure the des-hyperneat-rust submodule is initialized: "
                "git submodule update --init"
            )
        raw_features = []
        raw_classes = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                raw_classes.append(int(parts[0]))
                raw_features.append([float(x) for x in parts[1:]])

        # Min-max normalize each feature to [0, 1]
        features_array = np.array(raw_features)
        feat_min = features_array.min(axis=0)
        feat_max = features_array.max(axis=0)
        feat_range = feat_max - feat_min
        feat_range[feat_range == 0] = 1.0  # avoid division by zero
        normalized = (features_array - feat_min) / feat_range

        for i, class_id in enumerate(raw_classes):
            self.inputs.append(normalized[i].tolist() + [1.0])
            self.targets.append(self.CLASS_MAP[class_id])

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return list(zip(self.inputs, self.targets))


class DESRetinaProblem:
    """Retina problem matching DES-HyperNEAT specification (Tenstad & Glette 2021).

    8 binary pixel inputs + bias -> 2 pattern-match outputs.
    256 samples (16 left x 16 right patterns).
    Uses AHNI patterns via ESHyperNEATRetinaProblem 'independent' variant.
    """
    jitable = True
    use_bias = True
    fitness_threshold = 0.90

    def __init__(self):
        # Import here to avoid circular dependency
        import sys
        project_root = str(Path(__file__).resolve().parents[3] / 'src')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        raise NotImplementedError(  # geenns benchmark problem not bundled standalone
            "RetinaProblem requires the full geenns benchmark suite; it is not part "
            "of the standalone EMR-HyperNEAT release.")
        self._retina = ESHyperNEATRetinaProblem(variant='independent', use_bias=True)
        self.input_shape = self._retina.input_shape   # (9,)
        self.output_shape = self._retina.output_shape  # (2,)
        self.inputs = self._retina.inputs
        self.targets = self._retina.targets

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        return self._retina.get_data()


class TwoMoonsProblem:
    """Two Moons classification: two interleaving half-circles.

    Tests curved decision boundary that doesn't require oscillatory functions.
    Gaussian-style activations may outperform sine here.

    200 samples, 2D input + bias, binary classification.
    """
    jitable = True
    use_bias = True
    fitness_threshold = 0.90

    def __init__(self, n_samples: int = 200, noise: float = 0.10, seed: int = 42):
        self.n_samples = n_samples
        self.noise = noise
        self.seed = seed
        self.input_shape = (3,)  # x, y, bias
        self.output_shape = (1,)

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        rng = np.random.default_rng(self.seed)
        n_per = self.n_samples // 2

        # Upper moon (class 0)
        theta1 = np.linspace(0, np.pi, n_per)
        x1 = np.cos(theta1) + rng.normal(0, self.noise, n_per)
        y1 = np.sin(theta1) + rng.normal(0, self.noise, n_per)

        # Lower moon (class 1), shifted and flipped
        theta2 = np.linspace(0, np.pi, n_per)
        x2 = 1.0 - np.cos(theta2) + rng.normal(0, self.noise, n_per)
        y2 = -np.sin(theta2) + 0.5 + rng.normal(0, self.noise, n_per)

        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.array([0.0] * n_per + [1.0] * n_per)

        # Normalize to [-1, 1]
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X = 2.0 * (X - X_min) / (X_max - X_min + 1e-8) - 1.0

        # Shuffle
        idx = rng.permutation(len(X))
        data = []
        for i in idx:
            data.append((X[i].tolist() + [1.0], [y[i]]))
        return data


class GaussianXORProblem:
    """Gaussian XOR: 4 Gaussian clusters at XOR corners.

    Class 0 at (0,0) and (1,1), class 1 at (0,1) and (1,0).
    Tests whether aggregation-sensitivity transfers from Boolean XOR
    to continuous domains. Gating depth 1, encoding-limited.

    200 samples (50 per cluster), 2D input + bias, binary classification.
    """
    jitable = True
    use_bias = True
    fitness_threshold = 0.90

    def __init__(self, n_samples: int = 200, sigma: float = 0.15, seed: int = 42):
        self.n_samples = n_samples
        self.sigma = sigma
        self.seed = seed
        self.input_shape = (3,)  # x, y, bias
        self.output_shape = (1,)

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        rng = np.random.default_rng(self.seed)
        n_per = self.n_samples // 4

        centers = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]
        labels = [0.0, 0.0, 1.0, 1.0]  # XOR pattern

        X_all, y_all = [], []
        for (cx, cy), label in zip(centers, labels):
            x = rng.normal(cx, self.sigma, n_per)
            y = rng.normal(cy, self.sigma, n_per)
            X_all.append(np.column_stack([x, y]))
            y_all.extend([label] * n_per)

        X = np.vstack(X_all)
        y = np.array(y_all)

        # Normalize to [-1, 1]
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X = 2.0 * (X - X_min) / (X_max - X_min + 1e-8) - 1.0

        idx = rng.permutation(len(X))
        data = []
        for i in idx:
            data.append((X[i].tolist() + [1.0], [y[i]]))
        return data


class StripeProblem:
    """Stripe classification: 3 vertical stripes.

    Class 0 for |x| > 0.33 (left and right thirds),
    class 1 for |x| <= 0.33 (center third).
    Tests routing/thresholding computation (gating depth 0 but
    requires two decision boundaries).

    Prediction: aggregation-inert (routing, not gating).

    200 samples, 2D input + bias, binary classification.
    """
    jitable = True
    use_bias = True
    fitness_threshold = 0.90

    def __init__(self, n_samples: int = 200, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        self.input_shape = (3,)  # x, y, bias
        self.output_shape = (1,)

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        rng = np.random.default_rng(self.seed)

        # Uniform random points in [-1, 1] x [-1, 1]
        X = rng.uniform(-1.0, 1.0, (self.n_samples, 2))

        # Class 1 if center stripe (|x| <= 0.33), else class 0
        y = (np.abs(X[:, 0]) <= 0.33).astype(float)

        idx = rng.permutation(len(X))
        data = []
        for i in idx:
            data.append((X[i].tolist() + [1.0], [y[i]]))
        return data


class VisualDiscriminationProblem:
    """Visual Discrimination: large vs small square on a grid.

    Classic HyperNEAT benchmark exploiting geometric/spatial patterns.
    A 3×3 or 5×5 grid with a small square vs large square placed at
    random positions. Binary classification: large (1) vs small (0).

    Uses 5×5 grid = 25 binary pixel inputs + bias.
    """
    jitable = True
    use_bias = True
    fitness_threshold = 0.90

    def __init__(self, grid_size: int = 5, seed: int = 42):
        self.grid_size = grid_size
        self.seed = seed
        self.input_shape = (grid_size * grid_size + 1,)  # pixels + bias
        self.output_shape = (1,)

    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        rng = np.random.default_rng(self.seed)
        g = self.grid_size
        data = []

        # Generate all valid placements for small (1×1) and large (2×2) squares
        # Small square: any single pixel
        for r in range(g):
            for c in range(g):
                grid = np.zeros((g, g), dtype=float)
                grid[r, c] = 1.0
                data.append((grid.flatten().tolist() + [1.0], [0.0]))  # small = 0

        # Large square (2×2): any valid top-left corner
        for r in range(g - 1):
            for c in range(g - 1):
                grid = np.zeros((g, g), dtype=float)
                grid[r:r+2, c:c+2] = 1.0
                data.append((grid.flatten().tolist() + [1.0], [1.0]))  # large = 1

        # Shuffle deterministically
        idx = rng.permutation(len(data))
        data = [data[i] for i in idx]
        return data


def _get_mux_3():
    """Create a 3-bit multiplexer problem (1 address + 2 data bits)."""
    raise NotImplementedError(
        "MultiplexerProblem requires the full geenns benchmark suite; it is not "
        "bundled in the standalone EMR-HyperNEAT release.")


def _get_mux_6():
    """Create a 6-bit multiplexer problem (2 address + 4 data bits)."""
    raise NotImplementedError(
        "MultiplexerProblem requires the full geenns benchmark suite; it is not "
        "bundled in the standalone EMR-HyperNEAT release.")


# Problem registry for easy access
PROBLEMS = {
    'xor': XORProblem,
    'and': ANDProblem,
    'or': ORProblem,
    'nand': NANDProblem,
    'nor': NORProblem,
    'two_spirals': TwoSpiralsProblem,
    'step_function': StepFunctionProblem,
    'concentric_circles': ConcentricCirclesProblem,
    # Augmented coordinate variants
    'circles_augmented': AugmentedConcentricCirclesProblem,
    'circles_radius': RadiusOnlyCirclesProblem,
    'spirals_augmented': AugmentedTwoSpiralsProblem,
    # Continuous output reformulations
    'spirals_continuous': ContinuousTwoSpiralsProblem,
    # Polar coordinate variants
    'spirals_polar': PolarTwoSpiralsProblem,
    'spirals_polar_augmented': PolarAugmentedTwoSpiralsProblem,
    # Minimal feature variants
    'retina_counts': CountsRetinaProblem,
    # DES-HyperNEAT benchmark problems
    'iris': IrisProblem,
    'wine': WineProblem,
    'retina': DESRetinaProblem,
    # Compound Boolean tasks
    'xor_and_and': lambda: CompoundBooleanProblem('xor', 'and', 'and'),
    'xor_or_and': lambda: CompoundBooleanProblem('xor', 'and', 'or'),
    'and_xor_and': lambda: CompoundBooleanProblem('and', 'xor', 'and'),
    'xor_and_xor': lambda: CompoundBooleanProblem('xor', 'and', 'xor'),
    # Continuous 2D problems
    'two_moons': TwoMoonsProblem,
    'gaussian_xor': GaussianXORProblem,
    'stripe': StripeProblem,
    # Gaussian XOR sample-count variants (for neutral→sensitive transition sweep)
    'gaussian_xor_4': lambda: GaussianXORProblem(n_samples=4),
    'gaussian_xor_8': lambda: GaussianXORProblem(n_samples=8),
    'gaussian_xor_16': lambda: GaussianXORProblem(n_samples=16),
    'gaussian_xor_32': lambda: GaussianXORProblem(n_samples=32),
    'gaussian_xor_50': lambda: GaussianXORProblem(n_samples=50),
    'gaussian_xor_100': lambda: GaussianXORProblem(n_samples=100),
    'visual_discrimination': VisualDiscriminationProblem,
    # Multiplexer
    'mux_3': lambda: _get_mux_3(),
    'mux_6': lambda: _get_mux_6(),
    # Parity problems
    'parity_2': lambda: ParityProblem(n_bits=2),
    'parity_3': lambda: ParityProblem(n_bits=3),
    'parity_4': lambda: ParityProblem(n_bits=4),
    'parity_5': lambda: ParityProblem(n_bits=5),
    'parity_6': lambda: ParityProblem(n_bits=6),
    'parity_7': lambda: ParityProblem(n_bits=7),
    'parity_8': lambda: ParityProblem(n_bits=8),
    'parity_9': lambda: ParityProblem(n_bits=9),
    'parity_10': lambda: ParityProblem(n_bits=10),
}


def get_problem(name: str, **kwargs):
    """Get problem instance by name.

    Args:
        name: Problem name from PROBLEMS registry
        **kwargs: Problem-specific parameters

    Returns:
        Problem instance
    """
    if name not in PROBLEMS:
        raise ValueError(f"Unknown problem: {name}. Available: {list(PROBLEMS.keys())}")

    problem_class = PROBLEMS[name]
    if callable(problem_class) and not isinstance(problem_class, type):
        # It's a lambda
        return problem_class()
    return problem_class(**kwargs)


if __name__ == '__main__':
    # Test all problems
    print("Testing classification problems...")
    print()

    for name in PROBLEMS:
        problem = get_problem(name)
        data = problem.get_data()

        n_class_0 = sum(1 for _, t in data if t[0] == 0.0)
        n_class_1 = sum(1 for _, t in data if t[0] == 1.0)

        print(f"{name}:")
        print(f"  Input shape: {problem.input_shape}")
        print(f"  Output shape: {problem.output_shape}")
        print(f"  Data points: {len(data)}")
        print(f"  Class distribution: {n_class_0} / {n_class_1}")
        print()

    # Visualize Two Spirals
    try:
        import matplotlib.pyplot as plt

        problem = TwoSpiralsProblem()
        data = problem.get_data()

        x_0 = [d[0][0] for d in data if d[1][0] == 0.0]
        y_0 = [d[0][1] for d in data if d[1][0] == 0.0]
        x_1 = [d[0][0] for d in data if d[1][0] == 1.0]
        y_1 = [d[0][1] for d in data if d[1][0] == 1.0]

        plt.figure(figsize=(8, 8))
        plt.scatter(x_0, y_0, c='blue', label='Class 0', alpha=0.7)
        plt.scatter(x_1, y_1, c='red', label='Class 1', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Two Spirals Problem')
        plt.legend()
        plt.axis('equal')
        plt.savefig('/tmp/two_spirals.png', dpi=100)
        print("Saved visualization to /tmp/two_spirals.png")
    except ImportError:
        print("matplotlib not available, skipping visualization")
