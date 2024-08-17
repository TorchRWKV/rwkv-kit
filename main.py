import time
import os
import torch
from rwkvkit import RWKVConfig, rwkv6, sample_logits

if __name__ == '__main__':
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = rwkv6(
        model_path="weight/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth",
        state_path="weight/rwkv-0.pth",
        prefill_kernel="triton",
        use_jit=True,
        compile=True
    )


    # 设置续写的初始字符串和参数
    initial_string = """User: 怎么理解微积分？ \n\nAssistant: 我理解的微积分是一种数学分支，它研究的是函数的变化率和变化量，以及这些变化的关系。它在物理学、工程学、经济学等领域有广泛的应用，例如求导、积分、微分方程等。在数学中，微积分是一个非常重要的分支，它为我们解决各种问题提供了强大的工具。\n\nUser: 它的基本概念是什么？ \n\nAssistant:"""
    TEMPERATURE = 1.0  # 温度参数
    TOP_P = 0.0  # Top-p采样参数
    LENGTH_PER_TRIAL = 100  # 生成的长度

    print(model.generate(initial_string, LENGTH_PER_TRIAL, TEMPERATURE, TOP_P, include_prompt=True))
    print(model.chat([{"role": "user", "content": "你是什么模型?"}], 500, TEMPERATURE, TOP_P))
    for i in model.chat([{"role": "user", "content": "你好呀"}], 500, TEMPERATURE, TOP_P, stream=True):
        print(i, end="", flush=True)
