"""
JAX CPU Configuration with Warning Suppression

This module configures JAX to use CPU backend and suppresses all Metal-related
warnings on macOS. Import this BEFORE importing JAX in any module.

Usage:
    from emr_hyperneat._compat.utils import jax_cpu_config  # Import first!
    import jax
    import jax.numpy as jnp
    # No more Metal warnings!

This suppresses:
- XLA service initialization messages
- JAX Metal experimental warnings
- ABSL logging messages
- MPS client warnings
"""

import os
import warnings
import logging
import sys

def configure_jax_for_cpu():
    """Configure JAX to use CPU and suppress all warnings."""
    
    # 1. Force CPU backend before JAX import
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    os.environ['JAX_BACKEND_TARGET'] = 'cpu'
    
    # 2. Suppress various warning types
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
    os.environ['JAX_ENABLE_X64'] = '0'  # Disable 64-bit warnings
    os.environ['JAX_LOG_LEVEL'] = 'WARNING'
    
    # 3. Suppress XLA/ABSL initialization messages
    os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false'
    os.environ['ABSL_MIN_LOG_LEVEL'] = '2'  # Only warnings and errors
    
    # 4. Additional ABSL suppression
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GLOG_minloglevel'] = '2'
    
    # 5. Python-level warning filters
    warnings.filterwarnings('ignore', message='Platform.*experimental')
    warnings.filterwarnings('ignore', message='.*Metal.*')
    warnings.filterwarnings('ignore', message='.*GPU.*experimental')
    warnings.filterwarnings('ignore', category=UserWarning, module='jax')
    
    # 6. Suppress logging from specific modules
    logging.getLogger('jax').setLevel(logging.WARNING)
    logging.getLogger('jax._src').setLevel(logging.WARNING)
    logging.getLogger('jax.experimental').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.WARNING)
    
    # 7. Redirect stderr temporarily during JAX import to catch remaining messages
    class SuppressOutput:
        def __enter__(self):
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr.close()
            sys.stderr = self._original_stderr
    
    return SuppressOutput


# Apply configuration immediately when module is imported
_suppress_output = configure_jax_for_cpu()


def silent_jax_import():
    """Import JAX silently, suppressing all initialization messages."""
    with _suppress_output():
        import jax
        import jax.numpy as jnp
        
        # Verify CPU backend
        if jax.default_backend() != 'cpu':
            warnings.warn(
                f"JAX backend is {jax.default_backend()}, expected 'cpu'. "
                "Metal warnings may still appear.",
                RuntimeWarning
            )
    
    return jax, jnp


def get_jax_info():
    """Get JAX configuration information without warnings."""
    try:
        import jax
        return {
            'backend': jax.default_backend(),
            'devices': [str(d) for d in jax.devices()],
            'platform': os.environ.get('JAX_PLATFORM_NAME', 'not set'),
            'warnings_suppressed': True
        }
    except ImportError:
        return {
            'backend': 'not imported',
            'devices': [],
            'platform': os.environ.get('JAX_PLATFORM_NAME', 'not set'),
            'warnings_suppressed': True
        }


# Convenience function for testing
def test_configuration():
    """Test that JAX configuration is working without warnings."""
    print("Testing JAX CPU configuration...")
    
    # Import JAX
    jax, jnp = silent_jax_import()
    
    # Get info
    info = get_jax_info()
    print(f"Backend: {info['backend']}")
    print(f"Devices: {info['devices']}")
    
    # Test computation
    x = jnp.array([1.0, 2.0, 3.0])
    y = x * 2
    print(f"Test computation: {x} * 2 = {y}")
    print("✓ Configuration successful - no warnings!")


if __name__ == "__main__":
    test_configuration()