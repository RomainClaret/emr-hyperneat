"""Hardware information detection and reporting for experiments."""

import platform
import psutil
import subprocess
import json
from typing import Dict, Any, List, Optional
import os
import sys


class HardwareInfo:
    """Detects and reports hardware information for experiment tracking."""
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU information."""
        info = {
            'model': 'Unknown',
            'cores': psutil.cpu_count(logical=False) or 0,
            'threads': psutil.cpu_count(logical=True) or 0,
            'frequency': 'Unknown'
        }
        
        # Platform-specific CPU detection
        system = platform.system()
        
        if system == "Darwin":  # macOS
            try:
                # Get CPU brand
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, check=True
                )
                info['model'] = result.stdout.strip()
                
                # Get CPU frequency
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.cpufrequency_max'],
                    capture_output=True, text=True, check=True
                )
                freq_hz = int(result.stdout.strip())
                info['frequency'] = f"{freq_hz / 1e9:.2f} GHz"
            except:
                pass
                
        elif system == "Linux":
            try:
                # Parse /proc/cpuinfo
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            info['model'] = line.split(':')[1].strip()
                        elif 'cpu MHz' in line and info['frequency'] == 'Unknown':
                            mhz = float(line.split(':')[1].strip())
                            info['frequency'] = f"{mhz / 1000:.2f} GHz"
            except:
                pass
                
        elif system == "Windows":
            try:
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name,maxclockspeed'],
                    capture_output=True, text=True, check=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if parts:
                        info['model'] = ' '.join(parts[:-1])
                        info['frequency'] = f"{int(parts[-1]) / 1000:.2f} GHz"
            except:
                pass
        
        return info
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Get GPU information."""
        gpus = []
        
        # Try NVIDIA GPUs first
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        gpus.append({
                            'name': parts[0],
                            'memory': f"{parts[1]} MB",
                            'type': 'NVIDIA'
                        })
        except:
            pass
        
        # Check for Apple Silicon GPU
        if platform.system() == "Darwin" and platform.processor() == 'arm':
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPDisplaysDataType', '-json'],
                    capture_output=True, text=True, check=True
                )
                data = json.loads(result.stdout)
                for display in data.get('SPDisplaysDataType', []):
                    if 'sppci_model' in display:
                        gpus.append({
                            'name': display['sppci_model'],
                            'memory': display.get('spdisplays_vram', 'Unknown'),
                            'type': 'Apple Silicon'
                        })
            except:
                # Fallback for M-series chips
                if any(x in platform.machine() for x in ['arm64', 'aarch64']):
                    gpus.append({
                        'name': 'Apple Silicon GPU',
                        'memory': 'Unified Memory',
                        'type': 'Apple Silicon'
                    })
        
        return gpus
    
    @staticmethod
    def get_memory_info() -> Dict[str, str]:
        """Get system memory information."""
        vm = psutil.virtual_memory()
        return {
            'total': f"{vm.total / (1024**3):.1f} GB",
            'available': f"{vm.available / (1024**3):.1f} GB",
            'percent_used': f"{vm.percent:.1f}%"
        }
    
    @staticmethod
    def get_jax_info() -> Dict[str, Any]:
        """Get JAX backend information."""
        info = {
            'available': False,
            'backend': 'Unknown',
            'devices': []
        }
        
        try:
            import jax
            info['available'] = True
            info['version'] = jax.__version__
            
            # Get default backend
            backend = os.environ.get('JAX_PLATFORM_NAME', '').upper()
            if not backend:
                devices = jax.devices()
                if devices:
                    backend = devices[0].platform.upper()
            
            info['backend'] = backend or 'CPU'
            
            # Get device info
            for device in jax.devices():
                info['devices'].append({
                    'id': device.id,
                    'platform': device.platform,
                    'device_kind': getattr(device, 'device_kind', 'Unknown')
                })
                
            # Check Metal compatibility
            if platform.system() == "Darwin" and backend == "METAL":
                info['metal_status'] = "Experimental - Not all operations supported"
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    @staticmethod
    def get_platform_info() -> Dict[str, str]:
        """Get platform/OS information."""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    @staticmethod
    def get_full_hardware_info() -> Dict[str, Any]:
        """Get complete hardware information."""
        return {
            'platform': HardwareInfo.get_platform_info(),
            'cpu': HardwareInfo.get_cpu_info(),
            'memory': HardwareInfo.get_memory_info(),
            'gpus': HardwareInfo.get_gpu_info(),
            'jax': HardwareInfo.get_jax_info()
        }
    
    @staticmethod
    def format_for_report(info: Optional[Dict[str, Any]] = None) -> str:
        """Format hardware info for inclusion in experiment reports."""
        if info is None:
            info = HardwareInfo.get_full_hardware_info()
        
        lines = [
            "Hardware Information:",
            "=" * 50,
            f"Platform: {info['platform']['system']} {info['platform']['release']}",
            f"Python: {info['platform']['python_version']}",
            f"CPU: {info['cpu']['model']}",
            f"CPU Cores: {info['cpu']['cores']} physical, {info['cpu']['threads']} logical",
            f"Memory: {info['memory']['total']} ({info['memory']['percent_used']} used)",
        ]
        
        # GPU information
        if info['gpus']:
            lines.append(f"GPUs: {len(info['gpus'])}")
            for i, gpu in enumerate(info['gpus']):
                lines.append(f"  {i+1}. {gpu['name']} ({gpu['memory']})")
        else:
            lines.append("GPUs: None detected")
        
        # JAX information
        if info['jax']['available']:
            lines.append(f"JAX Backend: {info['jax']['backend']}")
            if 'metal_status' in info['jax']:
                lines.append(f"JAX Note: {info['jax']['metal_status']}")
        else:
            lines.append("JAX: Not available")
        
        return '\n'.join(lines)
    
    @staticmethod
    def get_experiment_tag() -> str:
        """Get a short tag identifying the hardware configuration."""
        info = HardwareInfo.get_full_hardware_info()
        
        # Create short identifier
        platform_short = {
            'Darwin': 'mac',
            'Linux': 'linux',
            'Windows': 'win'
        }.get(info['platform']['system'], 'unknown')
        
        # CPU identifier (simplified)
        cpu = info['cpu']['model'].lower()
        if 'apple' in cpu and 'm4' in cpu:
            cpu_short = 'm4'
        elif 'apple' in cpu and 'm3' in cpu:
            cpu_short = 'm3'
        elif 'apple' in cpu and 'm2' in cpu:
            cpu_short = 'm2'
        elif 'apple' in cpu and 'm1' in cpu:
            cpu_short = 'm1'
        elif 'intel' in cpu:
            cpu_short = 'intel'
        elif 'amd' in cpu:
            cpu_short = 'amd'
        else:
            cpu_short = 'cpu'
        
        # GPU identifier
        if info['gpus']:
            gpu = info['gpus'][0]['name'].lower()
            if 'rtx 2080' in gpu:
                gpu_short = 'rtx2080ti'
            elif 'rtx 3090' in gpu:
                gpu_short = 'rtx3090'
            elif 'rtx 4090' in gpu:
                gpu_short = 'rtx4090'
            elif 'apple' in gpu or info['gpus'][0]['type'] == 'Apple Silicon':
                gpu_short = 'metal'
            else:
                gpu_short = 'gpu'
        else:
            gpu_short = 'cpu_only'
        
        # JAX backend
        jax_backend = info['jax'].get('backend', 'none').lower()
        
        return f"{platform_short}_{cpu_short}_{gpu_short}_{jax_backend}"


class CacheInfo:
    """Detect CPU/GPU cache sizes for adaptive chunking optimization.

    This class provides runtime detection of cache hierarchy for use in
    performance-critical code that needs to adapt to different hardware.

    Supported platforms:
    - macOS (Apple Silicon and Intel): via sysctl
    - Linux: via /sys/devices/system/cpu/cpu0/cache/
    - GPU (NVIDIA): via nvidia-smi with known L2 cache lookup
    """

    @staticmethod
    def get_cpu_cache_info() -> Dict[str, int]:
        """Get CPU cache sizes in bytes.

        Returns:
            Dict with 'l1d', 'l2', 'l3' keys (sizes in bytes, 0 if unavailable)
        """
        system = platform.system()
        cache: Dict[str, int] = {'l1d': 0, 'l2': 0, 'l3': 0}

        if system == "Darwin":  # macOS
            try:
                # Apple Silicon has P-cores (perflevel0) and E-cores (perflevel1)
                # Use P-core cache sizes as they're typically used for compute
                p_core_keys = [
                    ('l1d', 'hw.perflevel0.l1dcachesize'),
                    ('l2', 'hw.perflevel0.l2cachesize'),
                ]
                fallback_keys = [
                    ('l1d', 'hw.l1dcachesize'),
                    ('l2', 'hw.l2cachesize'),
                    ('l3', 'hw.l3cachesize'),
                ]

                # Try P-core sizes first (Apple Silicon)
                for level, key in p_core_keys:
                    result = subprocess.run(
                        ['sysctl', '-n', key],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        cache[level] = int(result.stdout.strip())

                # Fall back to default keys (Intel Mac or if P-core query fails)
                if cache['l1d'] == 0:
                    for level, key in fallback_keys:
                        result = subprocess.run(
                            ['sysctl', '-n', key],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            cache[level] = int(result.stdout.strip())

                # Apple Silicon has no L3 - use SLC approximation (~48MB)
                # This is not directly queryable but is a reasonable estimate
                if cache['l3'] == 0 and cache['l2'] > 0:
                    # Estimate SLC based on chip variant
                    # M4 Max: ~48MB, M4 Pro: ~36MB, M4: ~16MB
                    # Use 3x L2 as SLC proxy (conservative for M4 Max)
                    cache['l3'] = cache['l2'] * 3
            except Exception:
                pass

        elif system == "Linux":
            try:
                # Read from /sys/devices/system/cpu/cpu0/cache/
                cache_path = "/sys/devices/system/cpu/cpu0/cache"
                if os.path.exists(cache_path):
                    for index_dir in os.listdir(cache_path):
                        index_path = os.path.join(cache_path, index_dir)
                        type_file = os.path.join(index_path, "type")
                        size_file = os.path.join(index_path, "size")
                        level_file = os.path.join(index_path, "level")

                        if os.path.exists(type_file) and os.path.exists(size_file):
                            with open(type_file) as f:
                                cache_type = f.read().strip()
                            with open(level_file) as f:
                                level = int(f.read().strip())
                            with open(size_file) as f:
                                size_str = f.read().strip()  # e.g., "32K", "12M"

                            # Parse size string
                            size_bytes = CacheInfo._parse_cache_size(size_str)

                            if level == 1 and cache_type == "Data":
                                cache['l1d'] = size_bytes
                            elif level == 2:
                                cache['l2'] = size_bytes
                            elif level == 3:
                                cache['l3'] = size_bytes
            except Exception:
                pass

        return cache

    @staticmethod
    def get_gpu_cache_info() -> Dict[str, Any]:
        """Get GPU L2 cache size if available.

        Returns:
            Dict with 'l2' (bytes), 'device_name', 'memory_total' (bytes)
        """
        info: Dict[str, Any] = {'l2': 0, 'device_name': '', 'memory_total': 0}

        try:
            # Try nvidia-smi for GPU info
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                # Handle multi-GPU systems: take first GPU line only
                first_line = result.stdout.strip().split('\n')[0]
                parts = first_line.split(', ')
                if len(parts) >= 2:
                    info['device_name'] = parts[0]
                    info['memory_total'] = int(parts[1]) * 1024 * 1024  # MB to bytes

                    # Known GPU L2 cache sizes (hardcoded lookup)
                    # nvidia-smi doesn't expose L2 cache directly
                    gpu_l2_cache = {
                        'A100': 40 * 1024 * 1024,      # 40 MB
                        'A10': 6 * 1024 * 1024,        # 6 MB
                        'V100': 6 * 1024 * 1024,       # 6 MB
                        'RTX 4090': 72 * 1024 * 1024,  # 72 MB
                        'RTX 4080': 64 * 1024 * 1024,  # 64 MB
                        'RTX 3090': 6 * 1024 * 1024,   # 6 MB
                        'RTX 3080': 5 * 1024 * 1024,   # 5 MB
                        'RTX 2080 Ti': 6 * 1024 * 1024,  # ~5.5 MB, using 6 MB
                        'RTX 2080': 4 * 1024 * 1024,   # 4 MB
                        'H100': 50 * 1024 * 1024,      # 50 MB
                        'L40': 48 * 1024 * 1024,       # 48 MB
                    }
                    for gpu_name, l2_size in gpu_l2_cache.items():
                        if gpu_name in info['device_name']:
                            info['l2'] = l2_size
                            break

                    # Default fallback for unknown GPUs
                    if info['l2'] == 0:
                        # Conservative estimate: 6 MB (common for older GPUs)
                        info['l2'] = 6 * 1024 * 1024
        except Exception:
            pass

        return info

    @staticmethod
    def _parse_cache_size(size_str: str) -> int:
        """Parse cache size string like '32K' or '12M' to bytes."""
        size_str = size_str.upper().strip()
        multipliers = {'K': 1024, 'M': 1024 * 1024, 'G': 1024 * 1024 * 1024}
        for suffix, mult in multipliers.items():
            if size_str.endswith(suffix):
                return int(size_str[:-1]) * mult
        return int(size_str)

    @staticmethod
    def get_full_cache_info() -> Dict[str, Any]:
        """Get complete cache information for CPU and GPU.

        Returns:
            Dict with 'cpu' and 'gpu' cache info
        """
        return {
            'cpu': CacheInfo.get_cpu_cache_info(),
            'gpu': CacheInfo.get_gpu_cache_info(),
        }

    @staticmethod
    def format_cache_report(info: Optional[Dict[str, Any]] = None) -> str:
        """Format cache info for human-readable report."""
        if info is None:
            info = CacheInfo.get_full_cache_info()

        lines = ["Cache Information:", "=" * 40]

        # CPU cache
        cpu = info.get('cpu', {})
        if cpu.get('l1d'):
            lines.append(f"CPU L1d: {cpu['l1d'] / 1024:.0f} KB")
        if cpu.get('l2'):
            lines.append(f"CPU L2:  {cpu['l2'] / (1024*1024):.1f} MB")
        if cpu.get('l3'):
            lines.append(f"CPU L3:  {cpu['l3'] / (1024*1024):.1f} MB")

        # GPU cache
        gpu = info.get('gpu', {})
        if gpu.get('device_name'):
            lines.append(f"GPU: {gpu['device_name']}")
            if gpu.get('l2'):
                lines.append(f"GPU L2: {gpu['l2'] / (1024*1024):.0f} MB")

        return '\n'.join(lines)


if __name__ == "__main__":
    # Test hardware detection
    info = HardwareInfo.get_full_hardware_info()
    print(HardwareInfo.format_for_report(info))
    print(f"\nHardware tag: {HardwareInfo.get_experiment_tag()}")
    print(f"\nRaw info:")
    print(json.dumps(info, indent=2))

    # Test cache detection
    print("\n" + "=" * 50)
    cache_info = CacheInfo.get_full_cache_info()
    print(CacheInfo.format_cache_report(cache_info))
    print(f"\nRaw cache info:")
    print(json.dumps(cache_info, indent=2))