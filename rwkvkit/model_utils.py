##########################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
##########################################################################

import torch
import types
import os
import gc
import math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from packaging import version

from dataclasses import dataclass, asdict
from typing import Optional, Literal
from importlib import resources


JITMODULE = nn.Module
JITSCRIPT = lambda x: x


@dataclass
class RWKVConfig:
    model_path: str
    vocab_size: Optional[int] = 65536
    device: Optional[str] = None
    onnx_opset_version: Optional[int] = 18
    chunk_size: int = 0
    state_path: Optional[str] = ""
    data_format: Optional[Literal['fp32', 'fp16', 'bf16']] = 'bf16'
    prefill_kernel: Literal['torch', 'triton',
                                     'triton-chunk', 'manual-torch'] = 'torch'
    init_model: Optional[bool] = False
    use_jit: Optional[bool] = True
    n_embd: Optional[int] = 2048
    n_layer: Optional[int] = 24
    head_size_a: Optional[int] = 64
    head_size_divisor: Optional[int] = 8
    vocab_file: Optional[str] = None
    use_jit: Optional[bool] = True

    def __post_init__(self):
        if self.device is None:
            self.check_available_device()
        if self.vocab_file is None:
            with resources.path('rwkvkit.assets', 'rwkv_vocab_v20230424.txt') as path:
                self.vocab_file = str(path)
        if not self.use_jit:
            os.environ['DISABLE_JIT'] = '1'
        if 'triton' in self.prefill_kernel:
            try:
                import triton
            except ImportError:
                raise ImportError(
                    "Please install triton to use the triton prefill kernel.")
        if self.use_jit:
            global JITMODULE, JITSCRIPT
            JITMODULE = torch.jit.ScriptModule if self.use_jit else nn.Module
            JITSCRIPT = torch.jit.script_method if self.use_jit else lambda x: x


    def check_available_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            try:
                import torch_musa
                if torch.musa.is_available():
                    self.device = 'musa'
                    return
            except ImportError:
                pass

            try:
                pytorch_version = torch.__version__.split('.')[:2]
                if int(pytorch_version[0]) > 2 or (
                        int(pytorch_version[0]) == 2 and int(pytorch_version[1]) >= 4):
                    pass
                else:
                    import intel_extension_for_pytorch as ipex
                if torch.xpu.is_available():
                    self.device = 'xpu'
                    return
            except ImportError:
                pass

            try:
                import torch_npu
                if torch.npu.is_available():
                    self.device = 'npu'
                    return
            except ImportError:
                pass

        self.device = 'cpu'


##########################################################################
# RWKV TIMEMix
##########################################################################


class RWKV_Tmix_x060(nn.Module):
    def __init__(self, config: RWKVConfig, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.head_size = self.config.head_size_a
        self.n_head = self.config.n_embd // self.head_size
        assert self.config.n_embd % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (self.config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - \
                (layer_id / self.config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.config.n_embd)
            for i in range(self.config.n_embd):
                ddd[0, 0, i] = i / self.config.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(self.config.n_embd, D_MIX_LORA * 5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(
                5, D_MIX_LORA, self.config.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(self.config.n_embd)
            for n in range(self.config.n_embd):
                decay_speed[n] = -6 + 5 * \
                    (n / (self.config.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(
                decay_speed.reshape(1, 1, self.config.n_embd))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(
                torch.zeros(self.config.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(
                D_DECAY_LORA, self.config.n_embd).uniform_(-0.01, 0.01))

            tmp = torch.zeros(self.config.n_embd)
            for n in range(self.config.n_embd):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * \
                    (1 - (n / (self.config.n_embd - 1))) + zigzag
            self.time_faaaa = nn.Parameter(
                tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(
            self.config.n_embd, self.config.n_embd, bias=False)
        self.key = nn.Linear(self.config.n_embd,
                             self.config.n_embd, bias=False)
        self.value = nn.Linear(
            self.config.n_embd, self.config.n_embd, bias=False)
        self.output = nn.Linear(
            self.config.n_embd, self.config.n_embd, bias=False)
        self.gate = nn.Linear(self.config.n_embd,
                              self.config.n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, self.config.n_embd, eps=(
            1e-5) * (self.config.head_size_divisor**2))

##########################################################################
# RWKV ChannelMix
##########################################################################


class RWKV_CMix_x060(nn.Module):
    def __init__(self, config: RWKVConfig, layer_id: int, dim_fnn: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.dim_ffn = dim_fnn

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - \
                (layer_id / self.config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.config.n_embd)
            for i in range(self.config.n_embd):
                ddd[0, 0, i] = i / self.config.n_embd
            self.time_maa_k = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(self.config.n_embd, self.dim_ffn, bias=False)
        self.receptance = nn.Linear(
            self.config.n_embd, self.config.n_embd, bias=False)
        self.value = nn.Linear(self.dim_ffn, self.config.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

##########################################################################
# RWKV Block
##########################################################################


class Block(nn.Module):
    def __init__(self, config: RWKVConfig, layer_id: int, dim_fnn: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(self.config.n_embd)
        self.ln2 = nn.LayerNorm(self.config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(self.config.n_embd)

        self.att = RWKV_Tmix_x060(config, layer_id)
        self.ffn = RWKV_CMix_x060(config, layer_id, dim_ffn)

    def forward(self, x):

        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

##########################################################################
# RWKV Model
##########################################################################


class RWKV_x060(nn.Module):
    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config
        self.config.n_embd = config.n_embd
        self.dim_ffn = int((self.config.n_embd * 3.5) // 32 * 32)
        assert self.config.n_embd % 32 == 0
        assert self.config.n_embd % 32 == 0
        assert self.dim_ffn % 32 == 0

        self.emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.blocks = nn.ModuleList(
            [Block(self.config, i, self.dim_ffn) for i in range(self.config.n_layer)])
        self.ln_out = nn.LayerNorm(self.config.n_embd)
        self.head = nn.Linear(self.config.n_embd,
                              self.config.vocab_size, bias=False)

        # !!! When you train RWKV from scratch, try my initialization for best performance !!!
        self.init_params()

    def forward(self, idx):

        x = self.emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        x = self.head(x)

        return x

    def init_params(self):
        m = self.state_dict()
        n_params = 0

        for n in self.state_dict():
            p = m[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or n.endswith(
                    '_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias'):
                if 'ln_x.weight' in n:
                    layer_scale = (
                        1 + int(n.split('.')[1])) / self.self.config.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                # !!! If you are using positional embedding, maybe it's better to remove block.0.ln0, and use default initialization for emb.weight instead of my uniform_(a=-1e-4, b=1e-4) !!!
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.self.config.vocab_size > self.self.config.n_embd:
                    scale = 0.5 * \
                        math.sqrt(self.self.config.vocab_size /
                                  self.self.config.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith('.weight')  # should always be true

                for kk in [".att.output.", ".ffn.value.", ".ffn.receptance."]:
                    if kk in n:
                        scale = 0
                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                m[n] = torch.empty((shape[0], shape[1]), device=p.device)
                if scale == 0:
                    nn.init.zeros_(m[n])
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            n_params += m[n].numel()

        # print('model params', n_params)
        gc.collect()
