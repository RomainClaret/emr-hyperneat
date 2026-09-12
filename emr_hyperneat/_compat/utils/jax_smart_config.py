"""
Smart JAX Configuration with Platform and Hardware Detection

This module intelligently configures JAX based on the detected platform and hardware:
- macOS: Forces CPU mode to suppress Metal warnings
- Linux with GPUs: Configures GPU memory allocation appropriately  
- Other systems: Uses sensible defaults

Usage:
    from emr_hyperneat._compat.utils import jax_smart_config  # Import before JAX!
    import jax
    import jax.numpy as jnp
    # JAX is now configured optimally for your hardware
"""

import os
import sys
import platform
import warnings
import logging
import subprocess
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class PlatformInfo:
    """Detected platform and hardware information."""
    def __init__(self):
        self.system = platform.system().lower()  # 'darwin', 'linux', 'windows'
        self.is_macos = self.system == 'darwin'
        self.is_linux = self.system == 'linux'
        self.is_windows = self.system == 'windows'
        
        # Hardware detection
        self.has_cuda = False
        self.cuda_devices = []
        self.gpu_memory_mb = []
        self.has_metal = False
        self.cpu_count = 1
        
        # Detect hardware
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect available hardware accelerators."""
        # CPU count
        try:
            import multiprocessing
            self.cpu_count = multiprocessing.cpu_count()
        except:
            self.cpu_count = 1
            
        # CUDA detection
        if self.is_linux or self.is_windows:
            self._detect_cuda()
            
        # Metal detection (macOS)
        if self.is_macos:
            self.has_metal = self._detect_metal()
    
    def _detect_cuda(self):
        """Detect CUDA devices and their memory."""
        try:
            # Try nvidia-smi first
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            memory_mb = int(parts[1].strip())
                            self.cuda_devices.append(name)
                            self.gpu_memory_mb.append(memory_mb)
                
                self.has_cuda = len(self.cuda_devices) > 0
                
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            # nvidia-smi not available or failed
            # Try alternative detection via CUDA_VISIBLE_DEVICES
            cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_devices and cuda_devices != '-1':
                # Assume CUDA is available
                self.has_cuda = True
                # Can't detect memory without nvidia-smi
    
    def _detect_metal(self) -> bool:
        """Detect Metal support on macOS."""
        try:
            # Check if Metal framework is available
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'Metal' in result.stdout
        except:
            # Assume Metal is available on modern macOS
            return True
    
    def __str__(self) -> str:
        """String representation of platform info."""
        info = [f"Platform: {self.system}"]
        info.append(f"CPUs: {self.cpu_count}")
        
        if self.has_cuda:
            info.append(f"CUDA GPUs: {len(self.cuda_devices)}")
            for i, (dev, mem) in enumerate(zip(self.cuda_devices, self.gpu_memory_mb)):
                info.append(f"  GPU {i}: {dev} ({mem} MB)")
        
        if self.has_metal:
            info.append("Metal: Available")
            
        return "\n".join(info)


def configure_jax_smart(precision: str = None):
    """Configure JAX intelligently based on platform and hardware.
    
    Args:
        precision: Optional precision setting ('float32' or 'float64').
                  If None, respects existing JAX_ENABLE_X64 setting.
    """
    
    # Detect platform and hardware
    platform_info = PlatformInfo()
    
    # Log detected configuration
    logger.info(f"Detected platform configuration:\n{platform_info}")
    
    # Apply platform-specific configuration
    if platform_info.is_macos:
        _configure_macos(platform_info)
    elif platform_info.is_linux:
        _configure_linux(platform_info)
    elif platform_info.is_windows:
        _configure_windows(platform_info)
    else:
        _configure_default(platform_info)
    
    # Configure common settings
    _configure_common(precision)
    
    # Return suppress context manager for backward compatibility
    return SuppressOutput


def _configure_macos(platform_info: PlatformInfo):
    """Configure JAX for macOS - suppress Metal warnings."""
    logger.info("Configuring JAX for macOS - forcing CPU mode to suppress Metal warnings")
    
    # Force CPU backend
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    os.environ['JAX_BACKEND_TARGET'] = 'cpu'
    
    # Suppress Metal-related warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['JAX_LOG_LEVEL'] = 'WARNING'
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GLOG_minloglevel'] = '2'
    
    # Python-level warning filters
    warnings.filterwarnings('ignore', message='Platform.*experimental')
    warnings.filterwarnings('ignore', message='.*Metal.*')
    warnings.filterwarnings('ignore', message='.*GPU.*experimental')
    warnings.filterwarnings('ignore', category=UserWarning, module='jax')


def _configure_linux(platform_info: PlatformInfo):
    """Configure JAX for Linux - optimize for available hardware."""
    
    if platform_info.has_cuda:
        # Configure for GPU
        logger.info("Configuring JAX for Linux with CUDA GPUs")
        
        # Allow JAX to use GPU
        if 'JAX_PLATFORM_NAME' not in os.environ:
            os.environ['JAX_PLATFORM_NAME'] = 'gpu'
            os.environ['JAX_BACKEND_TARGET'] = 'gpu'
        
        # Configure GPU memory allocation
        _configure_gpu_memory(platform_info)
        
    else:
        # Configure for CPU
        logger.info("Configuring JAX for Linux CPU-only")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['JAX_BACKEND_TARGET'] = 'cpu'
        
        # Configure CPU threading
        _configure_cpu_threading(platform_info)


def _configure_windows(platform_info: PlatformInfo):
    """Configure JAX for Windows."""
    
    if platform_info.has_cuda:
        logger.info("Configuring JAX for Windows with CUDA GPUs")
        
        # Allow JAX to use GPU
        if 'JAX_PLATFORM_NAME' not in os.environ:
            os.environ['JAX_PLATFORM_NAME'] = 'gpu'
            os.environ['JAX_BACKEND_TARGET'] = 'gpu'
        
        # Configure GPU memory allocation
        _configure_gpu_memory(platform_info)
    else:
        logger.info("Configuring JAX for Windows CPU-only")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['JAX_BACKEND_TARGET'] = 'cpu'
        
        # Configure CPU threading
        _configure_cpu_threading(platform_info)


def _configure_default(platform_info: PlatformInfo):
    """Default configuration for unknown platforms."""
    logger.info("Configuring JAX with default settings")
    
    # Default to CPU for safety
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    os.environ['JAX_BACKEND_TARGET'] = 'cpu'
    
    # Configure CPU threading
    _configure_cpu_threading(platform_info)


def _configure_gpu_memory(platform_info: PlatformInfo):
    """Configure GPU memory allocation to prevent OOM errors."""
    
    # Check if memory allocation is already configured
    if 'XLA_PYTHON_CLIENT_PREALLOCATE' in os.environ:
        logger.info(f"GPU memory preallocation already configured: "
                   f"XLA_PYTHON_CLIENT_PREALLOCATE={os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")
        return
        
    if 'XLA_PYTHON_CLIENT_MEM_FRACTION' in os.environ:
        logger.info(f"GPU memory fraction already configured: "
                   f"XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")
        return
    
    # Determine memory allocation strategy
    total_gpu_memory = sum(platform_info.gpu_memory_mb)
    num_gpus = len(platform_info.gpu_memory_mb)
    
    if total_gpu_memory == 0:
        # Can't detect memory, use conservative defaults
        logger.warning("Could not detect GPU memory, using conservative allocation")
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
        
    elif total_gpu_memory < 8192:  # Less than 8GB
        # Small GPU, disable preallocation
        logger.info("Small GPU detected, disabling memory preallocation")
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        
    elif total_gpu_memory < 16384:  # 8-16GB
        # Medium GPU, use 40% allocation
        logger.info(f"Medium GPU detected ({total_gpu_memory}MB), using 40% memory allocation")
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
        
    else:  # Large GPU
        # Use 30% allocation for large GPUs to leave room for other processes
        logger.info(f"Large GPU detected ({total_gpu_memory}MB), using 30% memory allocation")
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
    
    # Additional XLA flags for memory efficiency
    # Note: Only use flags that are known to be valid
    xla_flags = os.environ.get('XLA_FLAGS', '')
    
    # Add GPU-specific optimization flags if not already present
    # These help with memory efficiency
    if '--xla_gpu_deterministic_ops=true' not in xla_flags:
        xla_flags += ' --xla_gpu_deterministic_ops=true'
    
    os.environ['XLA_FLAGS'] = xla_flags.strip()
    
    logger.info(f"GPU memory configuration complete - GPUs: {num_gpus}, "
               f"Total memory: {total_gpu_memory}MB")


def _configure_cpu_threading(platform_info: PlatformInfo):
    """Configure CPU threading for optimal performance."""
    
    # Check if already configured
    if 'XLA_FLAGS' in os.environ and 'multi_thread_eigen' in os.environ['XLA_FLAGS']:
        return
        
    # Use half the cores to avoid oversubscription
    num_threads = max(1, platform_info.cpu_count // 2)
    
    # Set XLA flags for multi-threading
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_cpu_multi_thread_eigen=true' not in xla_flags:
        xla_flags += f' --xla_cpu_multi_thread_eigen=true'
    if f'intra_op_parallelism_threads=' not in xla_flags:
        xla_flags += f' intra_op_parallelism_threads={num_threads}'
    
    os.environ['XLA_FLAGS'] = xla_flags.strip()
    
    # Set other threading environment variables
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    logger.info(f"Configured CPU threading with {num_threads} threads")


def _configure_common(precision: str = None):
    """Configure common JAX settings.
    
    Args:
        precision: Optional precision setting ('float32' or 'float64').
                  If None, respects existing JAX_ENABLE_X64 setting.
    """
    
    # Handle precision configuration
    # Note: TensorNEAT requires 32-bit precision (JAX_ENABLE_X64='0')
    # We respect any pre-existing setting to allow TensorNEAT to force 32-bit
    if 'JAX_ENABLE_X64' not in os.environ:
        # No existing setting, use provided precision or default
        if precision == 'float64':
            os.environ['JAX_ENABLE_X64'] = 'true'
            logger.debug("JAX_ENABLE_X64 set to 'true' (float64 requested)")
        elif precision == 'float32':
            os.environ['JAX_ENABLE_X64'] = '0'
            logger.debug("JAX_ENABLE_X64 set to '0' (float32 requested)")
        else:
            # Default to float32 for performance
            os.environ['JAX_ENABLE_X64'] = '0'
            logger.debug("JAX_ENABLE_X64 set to '0' (default float32 for performance)")
    else:
        # Log what's already set
        current_value = os.environ['JAX_ENABLE_X64']
        logger.debug(f"JAX_ENABLE_X64 already set to '{current_value}' - respecting existing setting")
    
    # Suppress various warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['ABSL_MIN_LOG_LEVEL'] = os.environ.get('ABSL_MIN_LOG_LEVEL', '1')
    
    # Configure JAX to be less strict about dtype promotion
    # This helps with mixed float32/float64 operations
    import jax
    try:
        # Enable numpy-like dtype promotion rules
        jax.config.update('jax_numpy_dtype_promotion', 'standard')
    except:
        # Older JAX versions might not have this config
        pass
    
    # Configure logging levels
    logging.getLogger('jax').setLevel(logging.WARNING)
    logging.getLogger('jax._src').setLevel(logging.WARNING)
    logging.getLogger('jax.experimental').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.WARNING)


class SuppressOutput:
    """Context manager to suppress output during JAX import."""
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr


# Don't apply configuration immediately - let users call configure_jax_smart() explicitly
# This allows precision to be configured before JAX import
_suppress_output = SuppressOutput


def silent_jax_import():
    """Import JAX silently, suppressing initialization messages."""
    with _suppress_output():
        import jax
        import jax.numpy as jnp
    
    # Log actual backend being used
    backend = jax.default_backend()
    devices = jax.devices()
    logger.info(f"JAX initialized - Backend: {backend}, Devices: {len(devices)}")
    
    return jax, jnp


def get_jax_info() -> Dict[str, any]:
    """Get JAX configuration information."""
    try:
        import jax
        return {
            'backend': jax.default_backend(),
            'devices': [str(d) for d in jax.devices()],
            'platform': os.environ.get('JAX_PLATFORM_NAME', 'not set'),
            'mem_fraction': os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'default'),
            'preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'default'),
            'xla_flags': os.environ.get('XLA_FLAGS', 'none')
        }
    except ImportError:
        return {
            'backend': 'not imported',
            'devices': [],
            'platform': os.environ.get('JAX_PLATFORM_NAME', 'not set')
        }


def configure_with_precision(precision: str):
    """Configure JAX with a specific precision setting.
    
    This function allows external code to reconfigure JAX with a specific
    precision setting. Note that this must be called before importing JAX
    or any JAX-dependent libraries.
    
    Args:
        precision: Precision setting ('float32' or 'float64')
        
    Raises:
        ValueError: If precision is not 'float32' or 'float64'
    """
    if precision not in ['float32', 'float64']:
        raise ValueError(f"Invalid precision: {precision}. Must be 'float32' or 'float64'")
    
    # Clear existing JAX_ENABLE_X64 setting
    if 'JAX_ENABLE_X64' in os.environ:
        del os.environ['JAX_ENABLE_X64']
    
    # Reconfigure with specified precision
    configure_jax_smart(precision)
    logger.info(f"JAX reconfigured with precision: {precision}")


def test_configuration():
    """Test the smart JAX configuration."""
    print("Testing Smart JAX Configuration...")
    print("=" * 50)
    
    # Show platform info
    platform_info = PlatformInfo()
    print(platform_info)
    print("=" * 50)
    
    # Import JAX
    jax, jnp = silent_jax_import()
    
    # Get configuration info
    info = get_jax_info()
    print(f"JAX Backend: {info['backend']}")
    print(f"JAX Devices: {info['devices']}")
    print(f"Platform setting: {info['platform']}")
    print(f"Memory fraction: {info['mem_fraction']}")
    print(f"Preallocate: {info['preallocate']}")
    print(f"XLA flags: {info['xla_flags']}")
    
    # Test computation
    print("\nTesting computation...")
    x = jnp.array([1.0, 2.0, 3.0])
    y = x * 2
    print(f"Test: {x} * 2 = {y}")
    
    # Test on GPU if available
    if info['backend'] == 'gpu':
        print("\nTesting GPU computation...")
        # Create larger array to test GPU
        large_x = jnp.ones((1000, 1000))
        large_y = large_x @ large_x.T
        print(f"Matrix multiplication shape: {large_y.shape}")
        print(f"Sum: {jnp.sum(large_y)}")
    
    print("\n✓ Smart configuration successful!")


if __name__ == "__main__":
    test_configuration()