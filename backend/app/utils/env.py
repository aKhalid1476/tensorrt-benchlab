"""Environment metadata collection."""
import platform
import subprocess
from datetime import datetime
from typing import Optional

import torch

from ..schemas.bench import EnvironmentMetadata


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


def get_environment_metadata() -> EnvironmentMetadata:
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
            gpu_driver_version = pynvml.nvmlSystemGetDriverVersion()
            cuda_version = torch.version.cuda
        except:
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

    return EnvironmentMetadata(
        gpu_name=gpu_name,
        gpu_driver_version=gpu_driver_version,
        cuda_version=cuda_version,
        torch_version=torch.__version__,
        tensorrt_version=None,  # TODO: Get TensorRT version when available
        cpu_model=cpu_model,
        timestamp=datetime.now(),
        git_commit=get_git_commit(),
    )
