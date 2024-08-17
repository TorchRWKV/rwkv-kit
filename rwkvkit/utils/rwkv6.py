from einops import rearrange
from rwkvkit.model_utils import RWKV_x060, RWKVConfig, JITMODULE, JITSCRIPT
from typing import Tuple, Optional, List, Dict, Generator, Union
import torch.nn as nn
import torch
import os
from rwkvkit.rwkv_tokenizer import RWKV_TOKENIZER
from rwkvkit.sampler import sample_logits
from torch.utils.checkpoint import checkpoint




class RWKV_Block(JITMODULE):
    """
    RWKV模型的块结构。

    Args:
        block_w (dict): 权重字典。
        n_embd (int): 嵌入维度。
        n_head (int): 头数。
    """

    def __init__(
            self,
            block_w: dict,
            n_embd: int,
            n_head: int,
            config: RWKVConfig,
            onnx_opset=16,
            kernel_function=None):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.onnx_opset = onnx_opset
        self.config = config
        self.kernel_function = kernel_function

        # 初始化层归一化
        if self.onnx_opset >= 17:
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln1.weight = nn.Parameter(block_w['ln1.weight'])
            self.ln1.bias = nn.Parameter(block_w['ln1.bias'])
            self.ln2 = nn.LayerNorm(n_embd)
            self.ln2.weight = nn.Parameter(block_w['ln2.weight'])
            self.ln2.bias = nn.Parameter(block_w['ln2.bias'])
        else:
            self.ln1_weight = nn.Parameter(block_w['ln1.weight'])
            self.ln1_bias = nn.Parameter(block_w['ln1.bias'])
            self.ln2_weight = nn.Parameter(block_w['ln2.weight'])
            self.ln2_bias = nn.Parameter(block_w['ln2.bias'])

        # 初始化激活函数
        self.silu = nn.SiLU(inplace=False)

        # 初始化注意力参数
        self.att_time_maa_x = nn.Parameter(block_w['att.time_maa_x'])
        self.att_time_maa_w = nn.Parameter(block_w['att.time_maa_w'])
        self.att_time_maa_k = nn.Parameter(block_w['att.time_maa_k'])
        self.att_time_maa_v = nn.Parameter(block_w['att.time_maa_v'])
        self.att_time_maa_r = nn.Parameter(block_w['att.time_maa_r'])
        self.att_time_maa_g = nn.Parameter(block_w['att.time_maa_g'])
        self.att_time_maa_w1 = nn.Parameter(block_w['att.time_maa_w1'])
        self.att_time_maa_w2 = nn.Parameter(block_w['att.time_maa_w2'])
        self.att_time_decay = nn.Parameter(block_w['att.time_decay'])
        self.att_time_decay_w1 = nn.Parameter(block_w['att.time_decay_w1'])
        self.att_time_decay_w2 = nn.Parameter(block_w['att.time_decay_w2'])
        self.att_time_faaaa = nn.Parameter(block_w['att.time_faaaa'])
        self.att_receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_receptance.weight = nn.Parameter(
            block_w['att.receptance.weight'])
        self.att_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_key.weight = nn.Parameter(block_w['att.key.weight'])
        self.att_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_value.weight = nn.Parameter(block_w['att.value.weight'])
        self.att_output = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_output.weight = nn.Parameter(block_w['att.output.weight'])
        self.att_gate = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_gate.weight = nn.Parameter(block_w['att.gate.weight'])

        if self.onnx_opset >= 18:
            self.att_group_norm = nn.GroupNorm(
                num_groups=n_head, num_channels=n_embd, eps=1e-5, affine=True)
            self.att_group_norm.weight = nn.Parameter(
                block_w['att.ln_x.weight'])
            self.att_group_norm.bias = nn.Parameter(block_w['att.ln_x.bias'])
        else:
            self.att_group_norm_weight = nn.Parameter(
                block_w['att.ln_x.weight'])
            self.att_group_norm_bias = nn.Parameter(block_w['att.ln_x.bias'])

        # 初始化前馈参数
        self.ffn_time_maa_k = nn.Parameter(block_w['ffn.time_maa_k'])
        self.ffn_time_maa_r = nn.Parameter(block_w['ffn.time_maa_r'])
        self.ffn_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_key.weight = nn.Parameter(block_w['ffn.key.weight'])
        self.ffn_receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_receptance.weight = nn.Parameter(
            block_w['ffn.receptance.weight'])
        self.ffn_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_value.weight = nn.Parameter(block_w['ffn.value.weight'])

    @JITSCRIPT
    def manual_layer_norm(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            eps: float = 1e-5) -> torch.Tensor:
        """
        人工层归一化函数
        Args:
            x (torch.Tensor): 输入张量，形状为 [Batch, *]，* 表示任意维度。
            weight (torch.Tensor): 归一化的权重张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            bias (torch.Tensor): 归一化的偏置张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            eps (float): 用于数值稳定性的小值，防止除以零。
        Returns:
            torch.Tensor: 经过手动层归一化后的张量，形状与输入的 x 相同。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        x_scaled = x_normalized * weight  # .unsqueeze(-1) #这里会自动广播对齐
        x_shifted = x_scaled + bias  # .unsqueeze(-1)
        return x_shifted

    @JITSCRIPT
    def manual_group_norm(
            self,
            x: torch.Tensor,
            num_groups: int,
            weight: torch.Tensor,
            bias: torch.Tensor,
            eps: float = 1e-5) -> torch.Tensor:
        """
        人工组归一化函数。
        Args:
            x (torch.Tensor): 输入张量，形状为 [Batch, n_embd]。（或者[Batch*L, n_embd]）
            num_groups (int): 分组数，这里为 RWKV 的注意力头数。
            weight (torch.Tensor): 归一化的权重张量，形状为 [n_embd]。
            bias (torch.Tensor): 归一化的偏置张量，形状为 [n_embd]。
            eps (float): 用于数值稳定性的小值，防止除以零。
        Returns:
            torch.Tensor: 经过人工组归一化后的张量，形状与输入的 x 相同。
        """
        N, C = x.shape
        # if C % num_groups != 0:
        # raise ValueError("num_channels must be divisible by num_groups")
        # 加上这个会有无法推断静态图的警告
        channels_per_group = C // num_groups
        # 重塑x以便于分组
        x = x.view(N, num_groups, channels_per_group)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        x_normalized = x_normalized.view(N, C)
        # 应用权重和偏置
        x_scaled = x_normalized * weight
        x_shifted = x_scaled + bias
        return x_shifted

    @JITSCRIPT
    def channel_mixing(
            self,
            x: torch.Tensor,
            state: torch.Tensor,
            i: int) -> torch.Tensor:
        """
        通道混合函数。

        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, n_embd]。
            state (torch.Tensor): 时间状态张量，形状为[Batch, State Size, n_embd]。
            i (int): 时间索引。

        Returns:
            torch.Tensor: 混合后的张量，形状与输入的x相同。
        """
        i0 = (2 + self.head_size) * i
        sx = state[:, i0] - x  # 信息压缩到每一层的编号0位置
        state[:, i0] = x
        xk = torch.addcmul(x, sx, self.ffn_time_maa_k)
        xr = torch.addcmul(x, sx, self.ffn_time_maa_r)
        r = torch.sigmoid(self.ffn_receptance(xr))
        k = torch.relu(self.ffn_key(xk)).pow_(2)
        output = r * self.ffn_value(k)
        return output

    @JITSCRIPT
    def channel_mixing_parallel(
            self,
            x: torch.Tensor,
            state: torch.Tensor,
            i: int,
            training: bool) -> torch.Tensor:
        """
        并行通道混合函数
        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, L, n_embd]。
            state (torch.Tensor): 时间状态张量，形状为[Batch, State Size, n_embd]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 混合后的张量，形状与输入的x相同。
        """
        i0 = (2 + self.head_size) * i

        sx_lerp = torch.empty_like(x)
        sx_lerp[:, 0] = state[:, i0] - x[:, 0]

        # for l in range(1, L):
        #     sx_lerp[:, l] = x[:, l-1] - x[:, l]
        # 和上方等同，使用矩阵运算计算差值
        sx_lerp[:, 1:] = x[:, :-1] - x[:, 1:]

        state[:, i0] = x[:, -1]  # 这里把state赋值为最后一个输入

        xk = torch.addcmul(x, sx_lerp, self.ffn_time_maa_k)
        xr = torch.addcmul(x, sx_lerp, self.ffn_time_maa_r)

        r = torch.sigmoid(self.ffn_receptance(xr))  # [Batch, L, n_embd]
        k = torch.relu(
            self.ffn_key(xk)).pow_(2) if not training else torch.relu(
            self.ffn_key(xk)).pow(2)

        output = r * self.ffn_value(k)
        return output

    def time_mixing(
            self,
            x: torch.Tensor,
            state: torch.Tensor,
            i: int) -> torch.Tensor:
        """
        时间混合函数。

        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, n_embd]。
            state (torch.Tensor): 时间状态张量，形状为[Batch, State Size, n_embd]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 混合后的时间状态张量，形状与输入的state相同。
        """
        batch_size, H, S = x.size(0), self.n_head, self.head_size
        x, state, g = self.time_mixing_jit(x, state, i, batch_size, H, S)

        # 展平x并应用组归一化和门控
        if self.onnx_opset >= 18:
            x = self.time_mixing_jit2(x, g)
        else:
            x = x.flatten(start_dim=1)
            x = self.manual_group_norm(
                x,
                num_groups=H,
                weight=self.att_group_norm_weight,
                bias=self.att_group_norm_bias) * g
            x = self.att_output(x)
        # 应用输出层并返回结果
        return x

    @JITSCRIPT
    def time_mixing_jit(self, x: torch.Tensor, state: torch.Tensor, i: int,
                        batch_size: int, H: int, S: int):
        i1 = (2 + S) * i + 1  # i是block的编号

        sx = state[:, i1] - x
        state[:, i1] = x  # 信息压缩到每一层的编号1位置

        xxx = torch.addcmul(x, sx, self.att_time_maa_x)
        xxx = torch.tanh(xxx @ self.att_time_maa_w1).view(batch_size, 5, 1, -1)
        xxx = torch.matmul(xxx, self.att_time_maa_w2).view(batch_size, 5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=1)

        xw, xk, xv, xr, xg = torch.empty_like(x), torch.empty_like(
            x), torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
        torch.addcmul(x, sx, self.att_time_maa_w + mw, out=xw)
        torch.addcmul(x, sx, self.att_time_maa_k + mk, out=xk)
        torch.addcmul(x, sx, self.att_time_maa_v + mv, out=xv)
        torch.addcmul(x, sx, self.att_time_maa_r + mr, out=xr)
        torch.addcmul(x, sx, self.att_time_maa_g + mg, out=xg)

        w = (self.att_time_decay +
             (torch.tanh(xw @ self.att_time_decay_w1) @ self.att_time_decay_w2))

        # 计算注意力机制的权重
        w = -torch.exp(w.view(batch_size, H, S, 1))
        # 计算注意力机制的组件
        r = self.att_receptance(xr).view(batch_size, H, 1, S)
        k = self.att_key(xk).view(batch_size, H, S, 1)
        v = self.att_value(xv).view(batch_size, H, 1, S)
        g = self.silu(self.att_gate(xg))

        # 使用注意力机制更新状态
        s = state[:, (2 + S) * i + 2:(2 + S) * (i + 1),
                  :].view(batch_size, H, S, S)
        a = k @ v
        x = r @ torch.addcmul(s, self.att_time_faaaa, a)
        s = torch.addcmul(a, torch.exp(w), s)
        # 更新第i层STATE的注意力参数
        state[:, (2 + S) * i + 2:(2 + S) * (i + 1),
              :] = s.view(batch_size, S, -1)
        return x, state, g

    @JITSCRIPT
    def time_mixing_jit2(self, x: torch.Tensor, g):
        return self.att_output(self.att_group_norm(x.flatten(start_dim=1)) * g)

    def time_mixing_parallel(
            self,
            x: torch.Tensor,
            state: torch.Tensor,
            i: int,
            training: bool = False) -> torch.Tensor:
        """
        并行处理的时间混合函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, L, n_embd]。
            state (torch.Tensor): 时间状态张量，形状为[Batch, State Size, n_embd]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 混合后的时间状态张量，形状与输入的state相同。
        """
        batch_size, L, H, S = x.size(0), x.size(1), self.n_head, self.head_size
        r, w, k, v, g, state = self.time_mixing_parallel_jit1(
            x, state, i, batch_size, L, H, S)
        x, state = self.apply_time_mixxing_kernel(
            r, w, k, v, state, i, batch_size, L, H, S, backend=self.config.prefill_kernel, training=training)
        # 展平x并应用组归一化
        if self.onnx_opset >= 18:
            x = self.time_mixing_parallel_jit2(x, g, batch_size, L)
        else:
            x = x.flatten(start_dim=2).view(batch_size * L, -1)
            x = self.manual_group_norm(
                x, num_groups=H, weight=self.att_group_norm_weight, bias=self.att_group_norm_bias).view(
                batch_size, L, -1) * g  # 因为组归一化强制要求Channel维度在第二个维度
            x = self.att_output(x)

        return x

    @JITSCRIPT
    def time_mixing_parallel_jit1(
            self,
            x: torch.Tensor,
            state: torch.Tensor,
            i: int,
            batch_size: int,
            L: int,
            H: int,
            S: int):
        i1 = (2 + S) * i + 1
        # 初始化结果张量
        sx_lerp, xxx = torch.empty_like(x), torch.empty_like(x)

        # 计算初始插值
        sx_lerp[:, 0] = state[:, i1] - x[:, 0]
        sx_lerp[:, 1:] = x[:, :-1] - x[:, 1:]

        state[:, i1] = x[:, -1]  # 这里把state赋值为最后一个输入

        xxx = x + sx_lerp * self.att_time_maa_x  # torch.Size([B, L, n_embd])
        # att_time_maa_w1: [n_embd, 160]
        xxx = torch.tanh(
            xxx @ self.att_time_maa_w1).view(batch_size, L, 5, 1, -1)
        xxx = torch.matmul(xxx, self.att_time_maa_w2).view(
            batch_size, L, 5, -1)  # [Batch, L, 5, n_embd]

        mw, mk, mv, mr, mg = xxx.unbind(dim=2)  # [10, 100, n_embd]

        xw = torch.addcmul(x, sx_lerp, self.att_time_maa_w + mw)
        xk = torch.addcmul(x, sx_lerp, self.att_time_maa_k + mk)
        xv = torch.addcmul(x, sx_lerp, self.att_time_maa_v + mv)
        xr = torch.addcmul(x, sx_lerp, self.att_time_maa_r + mr)
        xg = torch.addcmul(x, sx_lerp, self.att_time_maa_g + mg)

        w = (self.att_time_decay +
             (torch.tanh(xw @ self.att_time_decay_w1) @ self.att_time_decay_w2))
        w = -torch.exp(w.view(batch_size, L, H, S, 1))

        r = self.att_receptance(xr)
        k = self.att_key(xk)
        v = self.att_value(xv)
        g = self.silu(self.att_gate(xg))  # [10, 100, n_embd]

        return r, w, k, v, g, state

    def apply_time_mixxing_kernel(self,
                                  r: torch.Tensor,
                                  w: torch.Tensor,
                                  k: torch.Tensor,
                                  v: torch.Tensor,
                                  state: torch.Tensor,
                                  i: int,
                                  batch_size: int,
                                  L: int,
                                  H: int,
                                  S: int,
                                  backend: str = "torch",
                                  training: bool = False) -> Tuple[torch.Tensor,
                                                                   torch.Tensor]:
        """
        Apply the time mixing kernel operation with support for multiple backend implementations.

        This function implements the core time mixing operation of the RWKV model,
        offering different backend options for computation:

        - "torch": Native PyTorch implementation. Uses built-in PyTorch operations,
                which may be slower on certain hardware.

        - "manual-torch": Manual implementation with PyTorch, using FP32 for backward pass.
                        Suitable for scenarios requiring precise gradients.

        - "triton": Triton backend implementation, computing in FP32.
                    Offers a good balance between performance and precision.

        - "triton-chunk": chunk rwkv Triton backend, computing at input precision.
                        Provides optimal performance but may introduce some precision loss.
                        Suitable for state fine-tuning but not recommended for full fine-tuning
                        due to potentially larger gradient errors.

        Args:
            r, w, k, v (torch.Tensor): Input tensors
            state (torch.Tensor): State tensor
            i (int): Current head index being processed
            batch_size (int): Batch size
            L (int): Sequence length
            H (int): Number of attention heads
            S (int): State size
            backend (str): Computation backend choice, default is "torch"

        Returns:
            tuple: (x, state)
                x (torch.Tensor): Output tensor
                state (torch.Tensor): Updated state tensor
        """
        s = state[:, (2 + S) * i + 2:(2 + S) * (i + 1)
                  ].view(batch_size, H, S, S)
        # we dont want to support cuda, since it is only supported by nvidia
        # and AMD
        if backend != "torch":
            u = self.att_time_faaaa.view(self.n_head, self.head_size)
            r = rearrange(r.squeeze(-1), 'b l (h d) -> b h l d', h=H)
            k = rearrange(k, 'b l (h d) -> b h l d', h=H)
            v = rearrange(v, 'b l (h d) -> b h l d', h=H)
            w = rearrange(w.squeeze(-1), 'b l h d -> b h l d', h=H)

            # Apply the chosen kernel function
            o, state_layer = self.kernel_function(
                r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=True, training=training)
            x = rearrange(o, 'b h l d -> b l (h d)')
            state[:, (2 + S) * i + 2:(2 + S) * (i + 1)
                  ] = state_layer.view(batch_size, S, -1)

        else:
            x, state_layer = self.native_torch_time_mixing_kernel(
                r, w, k, v, s, batch_size, L, H, S)
            state[:, (2 + S) * i + 2:(2 + S) * (i + 1)
                  ] = state_layer.view(batch_size, S, -1)
        return x, state

    @JITSCRIPT
    def native_torch_time_mixing_kernel(
            self,
            r: torch.Tensor,
            w: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            s: torch.Tensor,
            batch_size: int,
            L: int,
            H: int,
            S: int):
        a = k.view(batch_size, L, H, S, 1) @ v.view(batch_size,
                                                    L, H, 1, S)  # a: [batch_size, L, H, S, S]
        w = torch.exp(w)
        state_s = torch.zeros(batch_size, L + 1, H, S, S,
                              dtype=s.dtype, device=s.device)
        state_s[:, 0] = s
        for l in range(L):
            state_s[:, l + 1] = torch.addcmul(a[:, l], w[:, l], state_s[:, l])
        x = r.view(batch_size,
                   L,
                   H,
                   1,
                   S) @ torch.addcmul(state_s[:,
                                              :-1,
                                              :,
                                              :,
                                              :],
                                      self.att_time_faaaa,
                                      a)
        return x, state_s[:, -1].view(batch_size, S, -1)

    @JITSCRIPT
    def time_mixing_parallel_jit2(
            self,
            x: torch.Tensor,
            g: torch.Tensor,
            batch_size: int,
            L: int):
        return self.att_output(self.att_group_norm(x.flatten(
            start_dim=2).view(batch_size * L, -1)).view(batch_size, L, -1) * g)

    @torch.no_grad()
    def forward(
            self,
            x: torch.Tensor,
            state: torch.Tensor,
            i: int) -> torch.Tensor:
        """
        模型的前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, N_embd]。
            state (torch.Tensor): 隐藏状态张量，形状为[Batch, State Size, N_embd]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 前向传播结果张量，形状与输入的x相同。
        """
        if self.onnx_opset >= 17:
            x = x + self.time_mixing(self.ln1(x), state, i)
            x = x + self.channel_mixing(self.ln2(x), state, i)
        else:
            x = x + self.time_mixing(self.manual_layer_norm(x,
                                     self.ln1_weight, self.ln1_bias, 1e-5), state, i)
            x = x + self.channel_mixing(self.manual_layer_norm(x,
                                        self.ln2_weight, self.ln2_bias, 1e-5), state, i)
        return x

    def forward_prefill(
            self,
            x: torch.Tensor,
            state: torch.Tensor,
            i: int,
            training: bool = False) -> torch.Tensor:
        """
        模型的并行前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, L, N_embd]。
            state (torch.Tensor): 隐藏状态张量，形状为[Batch, State Size, N_embd]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 前向传播结果张量，形状与输入的x相同。
        """
        if self.onnx_opset >= 17:
            x = x + self.time_mixing_parallel(self.ln1(x), state, i, training)
            x = x + \
                self.channel_mixing_parallel(self.ln2(x), state, i, training)
        else:
            x = x + self.time_mixing_parallel(self.manual_layer_norm(
                x, self.ln1_weight, self.ln1_bias, 1e-5), state, i, training)
            x = x + self.channel_mixing_parallel(self.manual_layer_norm(
                x, self.ln2_weight, self.ln2_bias, 1e-5), state, i, training)
        return x


class RWKV6(JITMODULE):
    """
    RWKV模型的RNN结构。

    Args:
        args (dict): 参数字典。
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.tokenizer = RWKV_TOKENIZER(self.config.vocab_file)
        try:
            self.onnx_opset_version = int(self.config.onnx_opset_version)
        except BaseException:
            self.onnx_opset_version = 16  # 默认是最低的，op17版本才支持LayerNorm算子，op18版本才支持GroupNorm算子

        self.data_format = self.config.data_format
        assert self.data_format in ['fp32', 'fp16', 'bf16']
        assert self.config.prefill_kernel in [
            "torch", "triton", "triton-chunk", "manual-torch"]
        if self.config.prefill_kernel == "manual-torch":
            from rwkvkit.ops.rwkv6 import native_recurrent_rwkv6
            self.kernel_func = native_recurrent_rwkv6
        elif self.config.prefill_kernel == "triton":
            from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
            self.kernel_func = fused_recurrent_rwkv6
        elif self.config.prefill_kernel == "triton-chunk":
            from fla.ops.rwkv6 import chunk_rwkv6
            self.kernel_func = chunk_rwkv6
        else:
            self.kernel_func = None

        # 加载权重
        if self.config.init_model:
            self.init_params()
        else:
            self.load_params()

        self.eval()
        self._convert_dataformat()
        self.to(self.config.device)

    def _convert_dataformat(self):
        if self.data_format == 'fp16':
            self.half()
        elif self.data_format == 'fp32':
            self.float()
        elif self.data_format == 'bf16':
            self.bfloat16()

    def init_params(self):
        model_init = RWKV_x060(self.config)
        # 使用初始化的权重加载模型
        self.load_params(load_from_file=False, w=model_init.state_dict())
        del model_init
        import gc
        gc.collect()

    def load_params(self, load_from_file: bool = True, w: dict = None):
        if load_from_file:
            if not self.config.model_path.endswith('.pth'):
                self.config.model_path += '.pth'
            w = torch.load(self.config.model_path, map_location="cpu")
        else:
            assert w is not None

        # 将所有权重转换为float32
        self.num_layer = 0
        for k in w.keys():
            if '.time_' in k:
                w[k] = w[k].squeeze()
            if '.time_faaaa' in k:
                w[k] = w[k].unsqueeze(-1)
            if "blocks" in k:
                self.num_layer = max(self.num_layer, int(k.split(".")[1]))
        self.num_layer += 1

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.n_embd = w['blocks.0.ln1.weight'].shape[0]
        self.head_size = self.n_embd // self.n_head
        self.state_size = [self.num_layer * (2 + self.head_size), self.n_embd]

        # 初始化模型参数
        self.emb = nn.Embedding.from_pretrained(w['emb.weight'], freeze=True)

        if self.onnx_opset_version >= 17:
            self.ln0 = nn.LayerNorm(self.n_embd)
            self.ln0.weight = nn.Parameter(w['blocks.0.ln0.weight'])
            self.ln0.bias = nn.Parameter(w['blocks.0.ln0.bias'])
        else:
            self.ln0_weight = nn.Parameter(w['blocks.0.ln0.weight'])
            self.ln0_bias = nn.Parameter(w['blocks.0.ln0.bias'])

        self.blocks: List[RWKV_Block] = nn.ModuleList()

        for i in range(self.num_layer):
            # 提取当前块的权重
            block_w = {k[len(f'blocks.{i}.'):]: v for k,
                       v in w.items() if f'blocks.{i}.' in k}
            self.blocks.append(
                RWKV_Block(
                    block_w,
                    self.n_embd,
                    self.n_head,
                    self.config,
                    self.onnx_opset_version,
                    self.kernel_func))

        if self.onnx_opset_version >= 17:
            self.ln_out = nn.LayerNorm(self.n_embd)
            self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
            self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
        else:
            self.ln_out_weight = nn.Parameter(w['ln_out.weight'])
            self.ln_out_bias = nn.Parameter(w['ln_out.bias'])

        self.head = nn.Linear(self.n_embd, self.config.vocab_size, bias=False)
        self.head.weight = nn.Parameter(w['head.weight'])

    @JITSCRIPT
    def manual_layer_norm(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            eps: float = 1e-5) -> torch.Tensor:
        """
        人工层归一化函数
        Args:
            x (torch.Tensor): 输入张量，形状为 [Batch, *]，* 表示任意维度。
            weight (torch.Tensor): 归一化的权重张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            bias (torch.Tensor): 归一化的偏置张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            eps (float): 用于数值稳定性的小值，防止除以零。
        Returns:
            torch.Tensor: 经过手动层归一化后的张量，形状与输入的 x 相同。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        x_scaled = x_normalized * weight
        x_shifted = x_scaled + bias
        return x_shifted

    @torch.no_grad
    def forward(self,
                token: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
                                                               torch.Tensor]:
        if state is None:
            state = self.init_state(token.size(0))
        if token.dim() == 1:
            out, state = self.forward_autoregressive(token, state)
        elif self.config.chunk_size == 0:
            out, state = self.forward_prefill(token, state)
            out = out[:, -1]
        else:
            out, state = self.forward_prefill_chunks(
                token, state, self.config.chunk_size)
            out = out[:, -1]
        return out, state

    @torch.no_grad
    def chat(self,
             messages: List[Dict[str,
                                 str]],
             max_len: int = 512,
             temperature: float = 1.0,
             top_p: float = 0.6,
             end_id: int = 0,
             stream: bool = False) -> List[str]:
        """
        聊天函数。
        Args:
            messages (List[str]): 输入的消息列表。
            max_len (int): 最大生成长度。
        Returns:
            List[str]: 生成的回复列表。
        """
        prompt = self.apply_chat_temple(messages)
        return self.generate(
            prompt,
            max_len,
            temperature,
            top_p,
            end_id,
            include_prompt=False,
            stream=stream)

    def apply_chat_temple(self, messages: List[str]) -> str:
        prompt = ""
        for i in messages:
            if i["role"] == "user":
                prompt += f"User: {i['content']} \n\n"
            elif i['role'] == "system":
                prompt += f"System: {i['content']} \n\n"
            elif i['role'] == "assistant":
                prompt += f"Assistant: {i['content']} \n\n"
        prompt += "Assistant:"
        return prompt

    @torch.no_grad
    def generate(self,
                 prompt: str,
                 max_len: int = 512,
                 temperature: float = 1.0,
                 top_p: float = 0.6,
                 end_id: int = 0,
                 include_prompt=True,
                 stream: bool = False,
                 stop=['\n\nUser',
                       '<|endoftext|>']) -> Union[str,
                                                  Generator[str,
                                                            None,
                                                            None]]:
        state = self.init_state(1)
        end_tensor = torch.tensor(
            end_id,
            dtype=torch.long,
            device=self.config.device)
        token = torch.tensor(
            self.tokenizer.encode(
                [prompt]),
            dtype=torch.long,
            device=self.config.device)

        out, state = self.forward_prefill(token, state)
        token = sample_logits(out[:, -1], temperature, top_p)


        def token_generator(token: torch.Tensor, state: torch.Tensor):
            for i in range(max_len):
                out, state = self.forward_autoregressive(token, state)
                token = sample_logits(out, temperature, top_p)
                if token == end_tensor:
                    break
                yield token

        if stream:
            def stream_generator():
                yield self.tokenizer.decode(token.unsqueeze(1).cpu().tolist())[0]
                for t in token_generator(token, state):
                    yield self.tokenizer.decode(t.unsqueeze(1).cpu().tolist())[0]
            return stream_generator()
        else:
            token_response = torch.empty((1, max_len + 1), dtype=torch.long, device=self.config.device)
            token_response[:, 0] = token
            for i, t in enumerate(token_generator(token, state), start=1):
                token_response[:, i] = t
            token_response = token_response[:, :i + 1]
            response = self.tokenizer.decode(token_response.cpu().tolist())[0]
            for s in stop:
                response = response.split(s)[0]
            if include_prompt:
                response = prompt + response
            return response

    def forward_autoregressive(self,
                               token: torch.Tensor,
                               state: torch.Tensor) -> Tuple[torch.Tensor,
                                                             torch.Tensor]:
        """
        模型的前向传播。
        Args:
            token (torch.Tensor): 输入的令牌张量。[Batch_size]
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
        Returns:
            torch.Tensor: 模型输出。
        """
        if self.onnx_opset_version >= 17:
            x = self.forward_jit1(token)
        else:
            x = self.emb(token)
            x = self.manual_layer_norm(x, self.ln0_weight, self.ln0_bias, 1e-5)
        # 开始循环推理RWKV Block
        for i, block in enumerate(self.blocks):
            x = block(x, state, i)

        if self.onnx_opset_version >= 17:
            x = self.forward_jit2(x)
        else:
            x = self.manual_layer_norm(
                x, self.ln_out_weight, self.ln_out_bias, 1e-5)
            x = self.head(x)
        return x, state

    @staticmethod
    def forward_prefill_wrapper(block: RWKV_Block, x, state, i, training):
        return block.forward_prefill(x, state, i, training)

    def forward_prefill(self,
                        token: torch.Tensor,
                        state: torch.Tensor,
                        training: bool = False) -> Tuple[torch.Tensor,
                                                         torch.Tensor]:
        """
        模型的并行前向传播。
        Args:
            token (torch.Tensor): 输入的令牌张量。[Batch_size, L]
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
        Returns:
            torch.Tensor: 模型输出。
        """
        if self.onnx_opset_version >= 17:
            x = self.forward_jit1(token)
        else:
            x = self.emb(token)
            x = self.manual_layer_norm(x, self.ln0_weight, self.ln0_bias, 1e-5)
        # 开始循环推理RWKV Block
        if training:
            for i, block in enumerate(self.blocks):
                x = checkpoint(self.forward_prefill_wrapper, block, x, state, i, True, use_reentrant=False)
        else:
            for i, block in enumerate(self.blocks):
                x = block.forward_prefill(x, state, i, training)
        if self.onnx_opset_version >= 17:
            x = self.forward_jit2(x)
        else:
            x = self.manual_layer_norm(
                x, self.ln_out_weight, self.ln_out_bias, 1e-5)
            x = self.head(x)
        return x, state

    @JITSCRIPT
    def forward_jit1(self, token: torch.Tensor) -> torch.Tensor:
        return self.ln0(self.emb(token))

    @JITSCRIPT
    def forward_jit2(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_out(x))

    @torch.no_grad()
    def forward_prefill_chunks(self,
                               token: torch.Tensor,
                               state: torch.Tensor,
                               slice_len: int = 64) -> Tuple[torch.Tensor,
                                                             torch.Tensor]:
        """
        模型的分段并行前向传播，减少显存/内存使用。
        Args:
            token (torch.Tensor): 输入的令牌张量。[Batch_size, L]
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
        Returns:
            torch.Tensor: 模型输出。
        """
        data_len = token.shape[1]
        for i in range((data_len - 2) // slice_len + 1):
            start = i * slice_len
            end = min((i + 1) * slice_len, data_len)
            token_i = token[:, start:end]
            token_out, state = self.forward_prefill(token_i, state)

        return token_out, state

    def init_state(self, batch_size: int) -> torch.Tensor:
        """
        初始化状态。
        rgs:
            batch_size (int): 批次大小。
        Returns:
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd], device="cpu"
        """
        # 初始化状态
        state = torch.zeros(batch_size, self.state_size[0], self.state_size[1])

        # 这里把训练好的state加载进去
        if self.config.state_path != '':
            STATE = torch.load(self.config.state_path.replace(
                ".pth", "") + '.pth', map_location=torch.device("cpu"))
            head_size = self.head_size
            for i, (key, value) in enumerate(STATE.items()):
                state[:, ((2 + head_size) * i + 2):((2 + head_size) * (i + 1)),
                      :] = value.contiguous().permute(0, 2, 1).reshape(head_size, -1)

        if self.data_format == 'fp16':
            state = state.half()
        elif self.data_format == 'bf16':
            state = state.bfloat16()
        elif self.data_format == 'fp32':
            state = state.float()

        return state.to(self.config.device)

    def save_state(self, state: torch.Tensor, filename: str, bf16=True):
        """
        保存隐藏状态张量到文件。

        Args:
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
            filename (str): 保存文件的路径和名称。

        Returns:
            None
        """
        head_size = self.head_size
        n_head = self.n_head
        STATE = {}
        for i in range(self.num_layer):
            start = (2 + head_size) * i + 2
            end = (2 + head_size) * (i + 1)
            # 使用 detach() 创建一个新的张量
            layer_state = state[:, start:end, :].detach()
            batch_size, _, _ = layer_state.size()
            assert batch_size == 1, "保存状态时批次大小必须为1, 其他时候未验证"  # 我甚至不知道怎么写 :(
            STATE[f'blocks.{i}.att.time_state'] = layer_state.contiguous().view(
                n_head, head_size, head_size).permute(0, 1, 2)

        if bf16:
            for key in STATE.keys():
                STATE[key] = STATE[key].bfloat16()
        else:
            for key in STATE.keys():
                STATE[key] = STATE[key].float()
        torch.save(STATE, filename)

    def save_model(self, model_path, bf16=True):
        """
        将训练后的模型保存为 .pth 文件。
        Args:
            model_path (str): 要保存的模型路径。
        """
        # 创建一个空字典来存储模型权重
        state_dict = {}

        # 保存词嵌入层的权重
        state_dict['emb.weight'] = self.emb.weight.data

        # 保存 RWKV_RNN 的权重
        for name, param in self.named_parameters():
            if self.onnx_opset_version >= 17:
                if 'ln0' in name:
                    state_dict[name.replace(
                        'ln0.', 'blocks.0.ln0.')] = param.data
                if 'blocks' not in name:
                    state_dict[name] = param.data
            else:
                if 'ln0_weight' in name:
                    state_dict['blocks.0.ln0.weight'] = param.data
                elif 'ln0_bias' in name:
                    state_dict['blocks.0.ln0.bias'] = param.data
                elif 'ln_out_weight' in name:
                    state_dict['ln_out.weight'] = param.data
                elif 'ln_out_bias' in name:
                    state_dict['ln_out.bias'] = param.data
                elif 'blocks' not in name:
                    state_dict[name] = param.data

        # 保存 RWKV_Block 的权重
        for i, block in enumerate(self.blocks):
            for name, param in block.named_parameters():
                # 根据 ONNX opset 版本对权重名称进行调整
                if self.onnx_opset_version >= 18:
                    if name == 'att_group_norm.weight':
                        name = 'att.ln_x.weight'
                    elif name == 'att_group_norm.bias':
                        name = 'att.ln_x.bias'
                elif self.onnx_opset_version >= 17:
                    if name == 'ln1.weight':
                        name = 'ln1.weight'
                    elif name == 'ln1.bias':
                        name = 'ln1.bias'
                    elif name == 'ln2.weight':
                        name = 'ln2.weight'
                    elif name == 'ln2.bias':
                        name = 'ln2.bias'
                    elif name == 'att_group_norm_weight':
                        name = 'att.ln_x.weight'
                    elif name == 'att_group_norm_bias':
                        name = 'att.ln_x.bias'
                else:
                    if name == 'ln0_weight':
                        name = 'ln0.weight'
                    elif name == 'ln0_bias':
                        name = 'ln0.bias'
                    elif name == 'ln1_weight':
                        name = 'ln1.weight'
                    elif name == 'ln1_bias':
                        name = 'ln1.bias'
                    elif name == 'ln2_weight':
                        name = 'ln2.weight'
                    elif name == 'ln2_bias':
                        name = 'ln2.bias'
                    elif name == 'att_group_norm_weight':
                        name = 'att.ln_x.weight'
                    elif name == 'att_group_norm_bias':
                        name = 'att.ln_x.bias'

                if name.startswith('att_'):
                    # 将 'att_' 替换为 'att.'
                    name = 'att.' + name[4:]
                elif name.startswith('ffn_'):
                    name = 'ffn.' + name[4:]

                if '.time_faaaa' in name:
                    param_data = param.data
                elif '.time_' in name:
                    param_data = param.data.unsqueeze(-1)
                else:
                    param_data = param.data

                state_dict[f'blocks.{i}.{name}'] = param_data

        for name in state_dict:
            if '.time_maa_w1' in name or '.time_decay_w1' in name or '.time_decay_w2' in name or 'att.time_faaaa' in name:
                state_dict[name] = state_dict[name].view(
                    state_dict[name].shape[0], state_dict[name].shape[1])
            elif '.time_maa_w2' in name:
                state_dict[name] = state_dict[name].view(
                    state_dict[name].shape[0],
                    state_dict[name].shape[1],
                    state_dict[name].shape[2])
            elif 'att.time_maa_x' in name or 'att.time_maa_w' in name or 'att.time_maa_k' in name or 'att.time_maa_v' in name or 'att.time_maa_r' in name or 'att.time_maa_g' in name \
                    or 'ffn.time_maa_k' in name or 'ffn.time_maa_r' in name or 'time_decay' in name:
                state_dict[name] = state_dict[name].view(
                    1, 1, state_dict[name].shape[0])
            else:
                state_dict[name] = state_dict[name]

        if bf16:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].bfloat16()
        else:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].float()
        # 保存模型权重到 .pth 文件
        if not model_path.endswith('.pth'):
            model_path += '.pth'
        torch.save(state_dict, model_path)
        print(f"Model saved as {model_path}")
