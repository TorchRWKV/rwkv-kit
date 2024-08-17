import torch
from typing import Optional, Literal
from rwkvkit.model_utils import RWKVConfig


def rwkv6(
    model_path: str,
    vocab_size: Optional[int] = 65536,
    device: Optional[str] = None,
    chunk_size: int = 0,
    state_path: Optional[str] = "",
    data_format: Optional[Literal['fp32', 'fp16', 'bf16']] = 'bf16',
    prefill_kernel: Literal['torch', 'triton', 'triton-chunk', 'manual-torch'] = 'torch',
    init_model: Optional[bool] = False,
    use_jit: Optional[bool] = True,
    n_embd: Optional[int] = 2048,
    n_layer: Optional[int] = 24,
    head_size_a: Optional[int] = 64,
    head_size_divisor: Optional[int] = 8,
    vocab_file: Optional[str] = None,
    compile: Optional[bool] = False
):
    """
    Initialize and return an RWKV6 model with the specified configuration.

    Args:
        model_path (str): Path to the model weights file.
        vocab_size (int, optional): Size of the vocabulary. Defaults to 65536.
        device (str, optional): Device to run the model on. Defaults to None.
        onnx_opset_version (int, optional): ONNX opset version. Defaults to 18.
        chunk_size (int, optional): Chunk size for processing. Defaults to 0.
        state_path (str, optional): Path to the state file. Defaults to "".
        data_format (str, optional): Data format for the model. Defaults to 'bf16'.
        prefill_kernel (str, optional): Prefill kernel type. Defaults to 'torch'.
        init_model (bool, optional): Whether to initialize the model. Defaults to False.
        use_jit (bool, optional): Whether to use JIT compilation. Defaults to True.
        n_embd (int, optional): Embedding dimension. Defaults to 2048.
        n_layer (int, optional): Number of layers. Defaults to 24.
        head_size_a (int, optional): Head size A. Defaults to 64.
        head_size_divisor (int, optional): Head size divisor. Defaults to 8.
        vocab_file (str, optional): Path to the vocabulary file. Defaults to None.

    Returns:
        RWKV6: Initialized RWKV6 model.
    """
    config = RWKVConfig(
        model_path=model_path,
        vocab_size=vocab_size,
        device=device,
        chunk_size=chunk_size,
        state_path=state_path,
        data_format=data_format,
        prefill_kernel=prefill_kernel,
        init_model=init_model,
        use_jit=use_jit,
        n_embd=n_embd,
        n_layer=n_layer,
        head_size_a=head_size_a,
        head_size_divisor=head_size_divisor,
        vocab_file=vocab_file
    )
    from rwkvkit.utils.rwkv6 import RWKV6
    if compile:
        return torch.compile(RWKV6(config=config))
    return RWKV6(config=config)