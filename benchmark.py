import time
import os
import torch
from torchrwkv import RWKV6, RWKVConfig, sample_logits

if __name__ == '__main__':
    config = RWKVConfig(model_path='weight/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
                        prefill_kernel="triton-chunk",
                        use_jit=True)


    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV6(config=config)
    # Please do not use torch.compile, since JIT is on by default
    # Also, this will reduce the accuracy of the model by unknown reasons
    # model = torch.compile(model)

    batch_size = 1
    state = model.init_state(batch_size)  # 初始化状态
    prefill_lenth = 1024
    TEMPERATURE = 1.0
    TOP_P = 0.0
    LENGTH_PER_TRIAL = 200



    import numpy as np
    token_np = np.ones([batch_size, prefill_lenth], dtype=int)

    token = torch.tensor(token_np).long().to(config.device)  #
    for i in range(10):
        out, state_temp = model.forward(token, state)

    total_time = 0
    for i in range(10):
        t1 = time.time()
        out, state_temp = model.forward(token, state)
        t2 = time.time()
        elapsed_time = (t2 - t1) * 1000
        total_time += elapsed_time
    average_time = total_time / 10
    print(f"Time for prefill: batch_size={batch_size}, token_length={prefill_lenth}, time={average_time:.2f} ms")
    del out, state_temp

    state = model.init_state(batch_size)
    start_time = time.time()
    for step in range(LENGTH_PER_TRIAL):
        out, state = model.forward(token, state)
        token = sample_logits(out, TEMPERATURE, TOP_P)
    end_time = time.time()
    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL * batch_size
    speed = tokens_generated / total_time
    print(f"Token generation speed: {speed:.2f} tokens/second (prefill are included)")

    batch_size = 128
    initial_string = """User"""
    encoded_input = model.tokenizer.encode([initial_string] * batch_size)
    token = torch.tensor(encoded_input).long().to(config.device)
    del model
    import gc
    gc.collect()
    config = RWKVConfig(model_path='weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096',
                        state_path='weight/rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.pth',
                        prefill_kernel="torch",)
    model = RWKV6(config=config)
    state = model.init_state(batch_size)
    out, state = model.forward(token, state)
    token = sample_logits(out, TEMPERATURE, TOP_P)
    start_time = time.time()
    for step in range(LENGTH_PER_TRIAL):
        out, state = model.forward(token, state)
        token = sample_logits(out, TEMPERATURE, TOP_P)
    end_time = time.time()
    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL * batch_size
    speed = tokens_generated / total_time
    print(f"Token generation speed: {speed:.2f} tokens/second (prefill are not included)")



