# -*- coding: utf-8 -*-

import functools

import torch
import subprocess
import re
from functools import lru_cache
from packaging import version


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def require_version(version, hint):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version
            require_version(version, hint)
            return fn(ctx,
                      *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                      **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
        return wrapper
    return decorator


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)
    return wrapper


@lru_cache(maxsize=None)
def get_available_device():
    if torch.cuda.is_available():
        return 'cuda'

    try:
        if version.parse(torch.__version__) >= version.parse('2.4'):
            if torch.xpu.is_available():
                return 'xpu'
        else:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                return 'xpu'
    except ImportError:
        pass

    try:
        import torch_musa
        if torch.musa.is_available():
            return 'musa'
    except ImportError:
        pass

    try:
        import torch_npu
        if torch.npu.is_available():
            return 'npu'
    except ImportError:
        pass

    return 'cpu'


@lru_cache(maxsize=None)
def check_compute_capacity(device):
    if device == 'cuda':
        if torch.cuda.is_available():
            try:
                nvidia_smi = subprocess.check_output("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", shell=True)
                compute_cap = nvidia_smi.decode('utf-8').strip()
                compute_cap_major = int(compute_cap.split('.')[0])
                return compute_cap_major >= 8
            except BaseException:
                return False
        else:
            return False

    elif device == 'xpu':
        try:
            clinfo_output = subprocess.check_output("clinfo | grep 'Max size for global variable'", shell=True)
            clinfo_output = clinfo_output.decode('utf-8').strip()
            sizes = re.findall(r'(\d+) \((\d+)KiB\)', clinfo_output)
            for size in sizes:
                if int(size[1]) > 128:
                    return True
            return False
        except BaseException:
            return False

    elif device == 'musa':
        return False

    elif device == 'npu':
        return False

    else:
        return False


device_capacity = check_compute_capacity(get_available_device())


if version.parse(torch.__version__) >= version.parse('2.4'):
    from torch.amp import custom_fwd, custom_bwd

    def custom_fwd_wrapper(**kwargs):
        return custom_fwd(**kwargs)

    def custom_bwd_wrapper(**kwargs):
        return custom_bwd(**kwargs)

else:
    from torch.cuda.amp import custom_fwd, custom_bwd

    def custom_fwd_wrapper(**decorator_kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **func_kwargs):
                all_kwargs = {**decorator_kwargs, **func_kwargs}
                all_kwargs.pop('device_type', None)
                return custom_fwd(func)(*args, **all_kwargs)
            return wrapper
        return decorator

    def custom_bwd_wrapper(**decorator_kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **func_kwargs):
                all_kwargs = {**decorator_kwargs, **func_kwargs}
                all_kwargs.pop('device_type', None)
                return custom_bwd(func)(*args, **all_kwargs)
            return wrapper
        return decorator
