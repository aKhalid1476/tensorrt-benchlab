"""Environment metadata collection."""
import platform
import subprocess
import sys
from datetime import datetime
from typing import Optional

import torch
from contracts import EnvironmentMetadata


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return None


def get_environment_metadata(sanity_check_passed: Optional[bool] = None) -> EnvironmentMetadata:
    """Collect environment and system metadata."""
    gpu_name = None
    gpu_driver_version = None
    cuda_version = None

    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            gpu_driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(gpu_driver_version, bytes):
                gpu_driver_version = gpu_driver_version.decode('utf-8')
            cuda_version = torch.version.cuda
        except:
            pass

    # Get TensorRT version
    tensorrt_version = None
    try:
        import tensorrt as trt
        tensorrt_version = trt.__version__
    except ImportError:
        pass

    # Try to get CPU model
    cpu_model = None
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True
            )
            cpu_model = result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        break
    except:
        pass

    # Get Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return EnvironmentMetadata(
        gpu_name=gpu_name,
        gpu_driver_version=gpu_driver_version,
        cuda_version=cuda_version,
        torch_version=torch.__version__,
        tensorrt_version=tensorrt_version,
        python_version=python_version,
        cpu_model=cpu_model,
        timestamp=datetime.now(),
        git_commit=get_git_commit(),
        sanity_check_passed=sanity_check_passed,
    )
