import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional, Tuple


@torch.jit.script
def _sample_logits_as_single(
        out: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.8) -> torch.Tensor:
    """
    Sample from the logits tensor produced by the model.

    Args:
        out (torch.Tensor): Logits tensor from the model, shape [*, vocab_size].
        temperature (float): Temperature parameter for controlling the diversity of sampling. Default is 1.0.
        top_p (float): Top-p truncation parameter for stabilizing and controlling the sampling probability distribution. Default is 0.8.

    Returns:
        torch.Tensor: Sampled indices, shape [*]. For example, tensor([10464]).
    """
    assert temperature > 0, "Temperature should be positive"
    assert 0 <= top_p <= 1, "Top-p should be in the range [0, 1]"

    if top_p == 0.0:
        # Deterministically select the most likely token
        return torch.argmax(out, dim=-1)

    if top_p == 1.0:
        return torch.multinomial(
            torch.nn.functional.softmax(
                out, dim=-1), num_samples=1).squeeze(1)

    # Convert logits to log probabilities
    log_probabilities = torch.nn.functional.log_softmax(
        out / temperature, dim=-1)

    # Compute the cumulative log probabilities
    cumulative_log_probs = torch.cumsum(log_probabilities, dim=-1)

    # Create a mask to identify the tokens to remove based on top_p
    mask_remove = cumulative_log_probs > torch.log(
        torch.tensor(top_p, device=cumulative_log_probs.device))

    # Set the probabilities of tokens to remove to a very small value (e.g.,
    # -1e10)
    log_probabilities = log_probabilities.masked_fill(mask_remove, -1e10)

    # Generate a single sample
    sampled_index = torch.multinomial(
        torch.exp(log_probabilities), num_samples=1).squeeze(1)

    return sampled_index


@torch.jit.script
def _sample_logits_as_batch(
        out: torch.Tensor,
        temperature: torch.Tensor,
        top_p: torch.Tensor,
        sampled_indices: torch.Tensor) -> torch.Tensor:
    assert (temperature > 0).all(), "Temperature should be positive"
    assert (
        0 <= top_p).all() and (
        top_p <= 1).all(), "Top-p should be in the range [0, 1]"

    # Handle top_p == 0.0 and top_p == 1.0 cases
    argmax_mask = (top_p == 0.0)
    standard_mask = (top_p == 1.0)

    # Handle argmax case
    if argmax_mask.any():
        sampled_indices[argmax_mask] = torch.argmax(out[argmax_mask], dim=-1)

    # Handle standard sampling case
    if standard_mask.any():
        standard_probs = torch.nn.functional.softmax(
            out[standard_mask] / temperature[standard_mask].unsqueeze(1), dim=-1)
        sampled_indices[standard_mask] = torch.multinomial(
            standard_probs, num_samples=1).squeeze(1)

    # Handle top_p sampling for the rest
    top_p_mask = ~(argmax_mask | standard_mask)
    if top_p_mask.any():
        # Convert logits to log probabilities
        log_probs = torch.nn.functional.log_softmax(
            out[top_p_mask] / temperature[top_p_mask].unsqueeze(1), dim=-1)

        # Compute the cumulative probabilities
        sorted_logits, sorted_indices = torch.sort(
            log_probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.exp(sorted_logits), dim=-1)

        # Create a mask to identify the tokens to remove based on top_p
        remove_mask = cumulative_probs > top_p[top_p_mask].unsqueeze(1)
        remove_mask[:, 0] = False  # Always keep the top token

        # Set the probabilities of tokens to remove to a very small value
        sorted_logits[remove_mask] = -float('inf')

        # Sample from the filtered distribution
        probs = torch.exp(sorted_logits)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Map back to original vocab indices
        sampled_indices[top_p_mask] = torch.gather(
            sorted_indices, 1, sampled.unsqueeze(1)).squeeze(1)

    return sampled_indices


def sample_logits(
        out: torch.Tensor,
        temperature: float | torch.Tensor = 1.0,
        top_p: float | torch.Tensor = 0.8,
        use_cpu: bool = False) -> torch.Tensor:
    """
    Sample from the logits tensor produced by the model, with batch support for different temperature and top_p values.

    Args:
        out (torch.Tensor): Logits tensor from the model, shape [batch_size, vocab_size].
        temperature (torch.Tensor): Temperature parameter for each sample in the batch, shape [batch_size].
        top_p (torch.Tensor): Top-p parameter for each sample in the batch, shape [batch_size].

    Returns:
        torch.Tensor: Sampled indices, shape [batch_size].
    """
    # Convert temperature and top_p to tensors if they're floats
    if isinstance(temperature, float) and isinstance(top_p, float):
        return _sample_logits_as_single(out, temperature, top_p)
    else:
        batch_size, _ = out.shape
        device_o = out.device
        dtype = out.dtype
        # Ensure all tensors are on CPU
        if not isinstance(temperature, torch.Tensor):
            temperature = torch.full((batch_size,), float(
                temperature), dtype=dtype, device=out.device)
        if not isinstance(top_p, torch.Tensor):
            top_p = torch.full((batch_size,), float(
                top_p), dtype=dtype, device=out.device)

        if use_cpu:
            out, temperature, top_p = out.cpu(), temperature.cpu(), top_p.cpu()

        # Prepare results tensor
        sampled_indices = torch.zeros(
            batch_size, dtype=torch.long, device=out.device)
        return _sample_logits_as_batch(
            out, temperature, top_p, sampled_indices).to(device_o)


def apply_penalties(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    top_p: float | torch.Tensor,
    presence_penalty: float,
    frequency_penalty: float,
    token: Optional[torch.Tensor] = None,
    freq_dict: Optional[defaultdict] = None
) -> Tuple[torch.Tensor, torch.Tensor, defaultdict]:
    """
    Apply penalties to the logits tensor and sample from it.
    """
    if freq_dict is None:
        freq_dict = defaultdict(int)

    if token is not None:
        # 创建一个mask,标识token是否已经出现在生成的序列中
        mask = torch.zeros_like(logits, dtype=torch.bool).to(logits.device)
        mask[0][token.tolist()] = True

        # 对已出现的token施加存在惩罚
        logits = torch.where(mask, logits - presence_penalty, logits)

        # 根据token的出现频率施加频率惩罚
        freq_penalties = torch.tensor([freq_dict[i] for i in range(
            len(logits))], dtype=logits.dtype, device=logits.device)
        # 缩放频率惩罚,并裁剪到[0, 1]范围内
        freq_penalties = torch.clamp(freq_penalties / len(freq_dict), 0, 1)
        logits -= frequency_penalty * freq_penalties

    token_sampled = sample_logits(logits, temperature=temperature, top_p=top_p)

    # 更新频率字典
    freq_dict[token_sampled.item()] += 1

    if token is not None:
        token = torch.cat((token, token_sampled), 0)
    else:
        token = token_sampled

    return token_sampled, token, freq_dict

