"""
Experiment-level configuration schema.

Defines experiment control, performance, metrics, and infrastructure settings
that are not part of the core NEAT/HyperNEAT algorithm.

These parameters control:
- Experiment execution (generations, trials, seed, output)
- Performance optimization (JAX, multiprocessing, profiling)
- Metrics collection (tracking, telemetry, events)
- Checkpointing and persistence
- Infrastructure (Docker, SSH, Kubernetes)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from pathlib import Path


class PerformanceConfig(BaseModel):
    """Performance optimization configuration.

    Controls JAX compilation, multiprocessing, and performance profiling.
    """

    # JAX optimization
    use_jit: bool = Field(
        default=True,
        description="Enable JAX JIT compilation for performance"
    )
    use_pmap: bool = Field(
        default=False,
        description="Enable JAX pmap for parallel execution across devices"
    )
    jax_memory_fraction: float = Field(
        default=0.75,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to preallocate (0.1-1.0)"
    )
    jax_memory_tracking: Literal['skip', 'actual', 'rss_with_note'] = Field(
        default='skip',
        description="JAX memory tracking mode"
    )

    # Smart mutations
    use_smart_mutations: bool = Field(
        default=False,
        description="Enable gradient-guided mutations (experimental)"
    )
    use_jit_smart_mutations: bool = Field(
        default=False,
        description="JIT compile smart mutations (if enabled)"
    )

    # Multiprocessing
    use_multiprocessing: bool = Field(
        default=False,
        description="Enable multiprocessing for fitness evaluation"
    )
    n_processes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of worker processes (None = auto-detect)"
    )
    workers: int = Field(
        default=0,
        ge=0,
        description="Number of parallel workers (0 = auto)"
    )

    # Parallel evaluation
    parallel_evaluations: bool = Field(
        default=False,
        description="Evaluate genomes in parallel within generation"
    )
    num_genome_workers: int = Field(
        default=0,
        ge=0,
        description="Worker threads for parallel genome evaluation (0 = adaptive)"
    )
    parallel_genomes: bool = Field(
        default=False,
        description="Parallelize genome-level operations"
    )

    # Batching
    batch_evaluate: bool = Field(
        default=False,
        description="Batch fitness evaluations for GPU efficiency"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        description="Batch size for fitness evaluation"
    )

    # Profiling
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    profile_compilation: bool = Field(
        default=True,
        description="Profile JAX compilation overhead"
    )
    profile_memory: bool = Field(
        default=False,
        description="Profile memory usage"
    )
    enable_component_profiling: bool = Field(
        default=False,
        description="Enable ComponentProfiler for detailed performance analysis"
    )
    profiling_output_dir: str = Field(
        default='./profiles/tensorneat',
        description="Directory for profiling output files"
    )

    # Memory measurement
    measure_memory: bool = Field(
        default=False,
        description="Measure memory usage during execution"
    )
    measure_jit_overhead: bool = Field(
        default=False,
        description="Measure JIT compilation overhead"
    )
    memory_limit_mb: Optional[int] = Field(
        default=None,
        ge=100,
        description="Memory limit in MB (None = no limit)"
    )

    class Config:
        json_schema_extra = {
            "description": "Performance optimization and profiling configuration",
            "examples": [{
                "use_jit": True,
                "use_multiprocessing": False,
                "jax_memory_fraction": 0.75,
                "parallel_evaluations": True
            }]
        }


class MetricsConfig(BaseModel):
    """Metrics collection and tracking configuration.

    Controls what data is collected during evolution for analysis and debugging.
    """

    # Metrics levels
    substrate_metrics_level: Literal['NONE', 'BASIC', 'STANDARD', 'ADVANCED', 'FULL'] = Field(
        default='NONE',
        description="Level of substrate metrics to collect"
    )
    cppn_metrics_level: Literal['NONE', 'MINIMAL', 'STANDARD'] = Field(
        default='NONE',
        description="Level of CPPN metrics to collect"
    )
    tensorneat_jax_metric_level: Literal['none', 'minimal', 'standard', 'full'] = Field(
        default='minimal',
        description="TensorNEAT JAX metrics collection level"
    )
    lazy_metrics: bool = Field(
        default=True,
        description="Enable LazyMetricValue for deferred GPU↔CPU sync"
    )

    # Mutation tracking
    track_mutations: bool = Field(
        default=False,
        description="Track mutation events"
    )
    track_mutations_detailed: bool = Field(
        default=False,
        description="Track detailed mutation statistics"
    )
    track_mutations_analytics: bool = Field(
        default=False,
        description="Enable mutation analytics"
    )
    mutation_tracking_max_events: int = Field(
        default=10000,
        ge=100,
        description="Maximum mutation events to track"
    )

    # Innovation tracking
    track_innovations: bool = Field(
        default=False,
        description="Track innovation numbers"
    )
    track_innovations_detailed: bool = Field(
        default=False,
        description="Track detailed innovation statistics"
    )
    track_innovations_analytics: bool = Field(
        default=False,
        description="Enable innovation analytics"
    )
    innovation_metrics: Optional[List[Literal['dynamics', 'topology', 'diversity', 'fixation', 'age']]] = Field(
        default=None,
        description="Specific innovation metrics to track"
    )

    # Network tracking
    track_network_growth: bool = Field(
        default=True,
        description="Track network size growth over generations"
    )
    connection_log_frequency: int = Field(
        default=20,
        ge=1,
        description="Log connection stats every N generations"
    )

    # Real-time events
    enable_realtime_events: bool = Field(
        default=True,
        description="Enable real-time events.jsonl emission during execution"
    )
    batch_device_transfers: bool = Field(
        default=True,
        description="Batch GPU↔CPU transfers (reduces overhead, slight latency)"
    )
    event_emission_interval: int = Field(
        default=1,
        ge=1,
        description="Emit events every N generations (1 = every generation)"
    )

    # Telemetry
    enable_telemetry: bool = Field(
        default=False,
        description="Enable telemetry data collection"
    )
    telemetry_detailed_logging: bool = Field(
        default=False,
        description="Enable detailed telemetry logging"
    )
    telemetry_export_path: Optional[str] = Field(
        default=None,
        description="Path to export telemetry data"
    )

    class Config:
        json_schema_extra = {
            "description": "Metrics collection and tracking configuration",
            "examples": [{
                "substrate_metrics_level": "BASIC",
                "track_mutations": True,
                "enable_realtime_events": True,
                "event_emission_interval": 1
            }]
        }


class CheckpointConfig(BaseModel):
    """Checkpointing and persistence configuration.

    Controls saving of generation history, checkpoints, and visualizations.
    """

    # Checkpointing
    checkpoint_enabled: bool = Field(
        default=False,
        description="Enable experiment checkpointing"
    )
    checkpoint_interval: int = Field(
        default=10,
        ge=1,
        description="Save checkpoint every N generations"
    )
    checkpoint_dir: Optional[str] = Field(
        default=None,
        description="Directory for checkpoint files (None = auto)"
    )
    resume: bool = Field(
        default=False,
        description="Resume from latest checkpoint if available"
    )

    # History saving
    save_generation_history: bool = Field(
        default=True,
        description="Save full generation history (populations, fitness)"
    )
    generation_save_interval: int = Field(
        default=5,
        ge=1,
        description="Save generation data every N generations"
    )
    save_top_n_per_generation: int = Field(
        default=1,
        ge=1,
        description="Number of top genomes to save per generation"
    )
    compress_history: bool = Field(
        default=False,
        description="Compress generation history files"
    )

    # Visualization saving
    save_visualizations: bool = Field(
        default=False,
        description="Save network visualizations during evolution"
    )
    save_networks: bool = Field(
        default=True,
        description="Save best network topologies"
    )
    generate_visualization_images: bool = Field(
        default=False,
        description="Generate PNG/SVG images of networks"
    )

    class Config:
        json_schema_extra = {
            "description": "Checkpointing and persistence configuration",
            "examples": [{
                "checkpoint_enabled": True,
                "checkpoint_interval": 10,
                "save_generation_history": True,
                "save_visualizations": False
            }]
        }


class InfrastructureConfig(BaseModel):
    """Infrastructure and execution environment configuration.

    Controls subprocess isolation, Docker, and distributed execution.
    """

    # Subprocess isolation
    no_subprocess_isolation: bool = Field(
        default=False,
        description="Disable subprocess isolation (web interface control)"
    )
    subprocess_timeout: Optional[int] = Field(
        default=None,
        ge=60,
        description="Subprocess timeout in seconds (None = no timeout)"
    )
    restart_every_n_trials: Optional[int] = Field(
        default=None,
        ge=1,
        description="Restart subprocess every N trials (prevent memory leaks)"
    )
    max_subprocess_restarts: int = Field(
        default=3,
        ge=0,
        description="Maximum subprocess restart attempts"
    )

    # Docker
    infrastructure_id: Optional[str] = Field(
        default=None,
        description="Infrastructure to run on (local, docker-{uuid}, ssh-{uuid})"
    )
    docker_cpu_limit: Optional[float] = Field(
        default=None,
        ge=0.1,
        description="CPU limit for Docker experiments"
    )
    docker_memory_limit: Optional[str] = Field(
        default=None,
        description="Memory limit for Docker experiments (e.g., '4g')"
    )
    docker_isolated: bool = Field(
        default=True,
        description="True = sandboxed (results only), False = mount repo"
    )

    class Config:
        json_schema_extra = {
            "description": "Infrastructure and execution environment",
            "examples": [{
                "no_subprocess_isolation": False,
                "docker_isolated": True
            }]
        }


class OutputConfig(BaseModel):
    """Output and reporting configuration.

    Controls HTML export, dashboards, and result formatting.
    """

    # Output directories
    output_dir: Optional[str] = Field(
        default=None,
        description="Base directory for experiment outputs (None = auto)"
    )

    # HTML export
    export_html: bool = Field(
        default=True,
        description="Generate HTML report after experiment"
    )
    html_provider: Literal['unified', 'plotly', 'geenns_viz'] = Field(
        default='unified',
        description="HTML visualization provider"
    )
    no_html_optimization: bool = Field(
        default=False,
        description="Disable HTML optimization (larger files, faster generation)"
    )
    export_formats: Optional[List[str]] = Field(
        default=None,
        description="Additional export formats (e.g., ['pdf', 'json'])"
    )

    # Dashboard
    dashboard: bool = Field(
        default=False,
        description="Launch live dashboard during evolution"
    )
    dashboard_port: int = Field(
        default=8050,
        ge=1024,
        le=65535,
        description="Dashboard server port"
    )
    dashboard_host: str = Field(
        default='localhost',
        description="Dashboard server host"
    )
    dashboard_refresh_rate: int = Field(
        default=1,
        ge=1,
        description="Dashboard refresh interval in seconds"
    )

    class Config:
        json_schema_extra = {
            "description": "Output and reporting configuration",
            "examples": [{
                "export_html": True,
                "html_provider": "unified",
                "dashboard": False
            }]
        }


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration.

    Combines all experiment-level settings: execution control, performance,
    metrics, checkpointing, infrastructure, and output.
    """

    # Experiment identification
    name: Optional[str] = Field(
        default=None,
        description="Experiment name (None = auto-generated)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Experiment description"
    )

    # Evolution control (at experiment level, not algorithm level)
    generations: int = Field(
        default=100,
        ge=1,
        description="Maximum generations for each trial (overrides algorithm max_generations)"
    )
    trials: int = Field(
        default=1,
        ge=1,
        description="Number of independent trials to run"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Base random seed for experiment (trials use seed+trial_idx)"
    )

    # Execution control
    quick: bool = Field(
        default=False,
        description="Quick mode (reduced generations/population for testing)"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode (verbose logging, no optimization)"
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        ge=60,
        description="Maximum experiment duration in seconds (None = no timeout)"
    )

    # Component configurations
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance optimization configuration"
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics collection configuration"
    )
    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig,
        description="Checkpointing and persistence configuration"
    )
    infrastructure: InfrastructureConfig = Field(
        default_factory=InfrastructureConfig,
        description="Infrastructure and execution environment"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output and reporting configuration"
    )

    @field_validator('trials')
    @classmethod
    def validate_trials(cls, v: int) -> int:
        """Ensure trials count is reasonable."""
        if v > 1000:
            raise ValueError("Trials > 1000 is likely impractical")
        return v

    class Config:
        json_schema_extra = {
            "title": "Experiment Configuration",
            "description": "Complete experiment-level configuration",
            "examples": [{
                "name": "xor_hyperneat_baseline",
                "generations": 100,
                "trials": 5,
                "seed": 42,
                "performance": {
                    "use_jit": True,
                    "parallel_evaluations": True
                },
                "metrics": {
                    "substrate_metrics_level": "BASIC",
                    "enable_realtime_events": True
                },
                "checkpoint": {
                    "checkpoint_enabled": True,
                    "checkpoint_interval": 10
                },
                "output": {
                    "export_html": True,
                    "dashboard": False
                }
            }]
        }
