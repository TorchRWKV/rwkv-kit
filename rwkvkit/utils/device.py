import importlib
import importlib.metadata
import os
import warnings
from functools import lru_cache

import torch
import platform
from packaging import version

def is_ipex_available():
    "Checks if ipex is installed."

    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True



@lru_cache
def is_musa_available(check_device=False):
    "Checks if `torch_musa` is installed and potentially if a MUSA is in the environment"
    if importlib.util.find_spec("torch_musa") is None:
        return False

    import torch_musa  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no MUSA is found
            _ = torch.musa.device_count()
            return torch.musa.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "musa") and torch.musa.is_available()


@lru_cache
def is_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch_npu") is None:
        return False

    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache
def is_xpu_available(check_device=False):
    """
    Checks if XPU acceleration is available either via `intel_extension_for_pytorch` or via stock PyTorch (>=2.4) and
    potentially if a XPU is in the environment
    """

    "check if user disables it explicitly"

    if is_ipex_available():
        if not check_pytorch_version("1.12"):
            return False
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError:
            return False
    else:
        if not check_pytorch_version("2.3"):
            return False

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()

@lru_cache
def is_cuda_available(check_device=False):
    "Checks if CUDA is available and potentially if a GPU is in the environment"
    if not torch.cuda.is_available():
        return False
    if check_device:
        # Will raise a RuntimeError if no GPU is found
        try:
            _ = torch.cuda.device_count()
            return torch.cuda.is_available()
        except RuntimeError:
            return False

    return hasattr(torch, "cuda") and torch.cuda.is_available()


@lru_cache
def is_directml_available(check_device=False):
    """Checks if `torch_directml` is installed and potentially if DirectML is available in the environment"""
    if importlib.util.find_spec("torch_directml") is None:
        return False

    try:
        import torch_directml  # noqa: F401

        if check_device:
            try:
                device = torch_directml.device(torch_directml.default_device())
                torch.ones(1).to(device)
                return True, device
            except RuntimeError:
                return False, None
        return True, torch_directml.device(torch_directml.default_device())
    except ImportError:
        return False, None


@lru_cache(maxsize=None)
def check_pytorch_version(version_s:str):
    if version.parse(torch.__version__) >= version.parse(version_s):
        return True
    else:
        return False



def is_linux():
    return platform.system().lower() == 'linux'