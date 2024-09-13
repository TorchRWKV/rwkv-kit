# rwkv-kit

rwkv-kit 是一个纯 PyTorch 实现的 RWKV 大语言模型推理框架。该项目旨在为 RWKV x060 模型提供一个灵活、易于扩展的 PyTorch 实现，同时支持多种功能，如批量推理、并行推理、ONNX 格式导出和单机训练。

## 特性

- 原生 PyTorch 实现
- 支持批量推理
- 支持并行推理，充分发挥 RWKV 优势
- 代码整洁，易于阅读和二次开发
- 支持导出并推理 ONNX 格式模型
- 简单的单机训练

## 硬件支持

我们支持多种硬件设备，包括但不限于：
- NVIDIA GPU
- Intel GPU
- AMD GPU
- 摩尔线程 MUSA GPU
- 华为昇腾 NPU

欢迎贡献其他设备的支持。

## 安装和使用

1. 克隆仓库：
   ```
   git clone -b dev https://github.com/TorchRWKV/rwkv-kit.git
   ```

2. 安装依赖：
   ```
   cd rwkv-kit
   pip install -r requirements.txt
   # 如果您想用fla kernel，需要安装好triton和rwkv-fla
   pip install rwkv-fla[cuda] # pip install rwkv-fla[xpu], pip install rwkv-fla[rocm]
   ```

3. 从 [BlinkDL/rwkv-6-world](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main) 下载 RWKV6 模型，并将权重放置在 `weight` 文件夹中。

4. 修改 `main.py` 中的 `MODEL_NAME` 参数。

For normal use:
```
   python main.py
   ```

## ONNX 导出

1. 修改 `onnx_export.py` 中的参数以适配你想导出的模型。
2. 运行：
   ```
   python onnx_export.py
   ```
3. （可选）创建简化模型的目录：
   ```
   mkdir ONNX_Simplified
   ```
Benchmark: (we use native torch to autoregress)
```
    import time
    import os
    import torch
    from rwkvkit import rwkv6, sample_logits

    initial_string = """hello"""
    batch_size = 128
    TEMPERATURE = 1.0
    TOP_P = 0.0
    LENGTH_PER_TRIAL = 100
    model = rwkv6(
        model_path="weight/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth",
        prefill_kernel="torch", # torch, torch-manual, triton, triton-chunk
        use_jit=True,
        compile=False
    )
    state = model.init_state(batch_size)


    encoded_input = model.tokenizer.encode([initial_string] * batch_size)

    token = torch.tensor(encoded_input).long().to(model.device)  #
    state = None
    out, state = model.forward(token, state)
    for step in range(LENGTH_PER_TRIAL):
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P)
        out, state = model.forward(token_sampled, state)

    t1 = time.time()
    state = None
    out, state = model.forward(token, state)
    t2 = time.time()
    print(f"Time: {t2 - t1}")

    start_time = time.time()

    for step in range(LENGTH_PER_TRIAL):
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P)
        out, state = model.forward(token_sampled, state)


    end_time = time.time()
    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL * batch_size
    speed = tokens_generated / total_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Token generation speed: {speed:.2f} tokens/second")

```



| 方法 | 批次大小 | 令牌长度 | 预填充时间 (ms) | 令牌生成速度 (tokens/second) | 备注 |
|------|---------|----------|----------------|----------------------------|------|
| triton-chunk | 1 | 1024 | 132.50 | 42.83 | 速度最快，适合推理和训练 |
| triton | 1 | 1024 | 105.49 | - | 适合推理和训练, 高精度, 某些情况不如 chunk 的速度 |
| torch | 1 | 1024 | 595.22 | - | 适合在Triton不能使用的设备下推理 |
| torch-manual | 1 | 1024 | 2468.00 | - | 适合在Triton不能使用的设备下训练，高精度 |
| - | 1 | - | - | 48.42 | 不包含预填充 |
| - | 64 | - | - | 1266.77 | 不包含预填充 |
| - | 128 | - | - | 1875.03 | 不包含预填充 |

注意：
- "-" 表示数据未提供或不适用。
- 对于批次大小为 1 的情况，只有 triton-chunk 方法提供了令牌生成速度（包含预填充）。
- 对于其他批次大小，令牌生成速度不包含预填充时间。
- 在 WSL2，Pytorch 2.5， Intel Arc A770, 1.6B下测试

For normal use:
```
    initial_string = """User: 你好！ 请问你是什么模型？"""
    batch_size = 2
    state = None
    TEMPERATURE = 1.0
    TOP_P = 0.0
    LENGTH_PER_TRIAL = 100


    encoded_input = model.tokenizer.encode([initial_string] * batch_size)

    token = torch.tensor(encoded_input).long().to(config.device)
    token_all = torch.tensor(encoded_input).long().to(config.device)


    for step in range(LENGTH_PER_TRIAL):
        out, state = model.forward(token, state)
        token = sample_logits(out, TEMPERATURE, TOP_P)
        token_all = torch.cat((token_all, token.unsqueeze(1)), 1)

        os.system('cls' if os.name == 'nt' else 'clear')
        decoded_sequences = model.tokenizer.decode(token_all.cpu().tolist())
        for i, seq in enumerate(decoded_sequences):
            print(f"Batch {i+1}: {seq}")

```

你也可以尝试新的高级API:
```
    print(model.generate(initial_string, LENGTH_PER_TRIAL, TEMPERATURE, TOP_P, include_prompt=True))
    print(model.chat([{"role": "user", "content": "你是什么模型?"}], 500, TEMPERATURE, TOP_P))
    for i in model.chat([{"role": "user", "content": "你好呀"}], 500, TEMPERATURE, TOP_P, stream=True):
        print(i, end="", flush=True)
```

本地 OpenAI 兼容客户端:
```
python -m rwkv-kit.openai_server --model model_path --state state_path(optional) --host 0.0.0.0 --port 8848
```
## 注意

本框架目前仅支持 RWKV v6 模型，具体版本号为 x060。

## 未来计划

我们计划基于本项目适配香橙派推出的 AI Pro 开发板，实现在昇腾的生态上推理国产大语言模型 RWKV。

## 致谢

特别感谢：
- Yang, S., & Zhang, Y. (2024). FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism (Version 0.0.1) [Computer software]. https://github.com/sustcsonglin/flash-linear-attention

- [onnxsim_large_model](https://github.com/luchangli03/onnxsim_large_model.git)

我们在不同的内核中使用了基于他们工作的修改实现。

## 贡献

我们欢迎社区的贡献。请随时提交 PR 和提出 Issue。你的输入对改进项目很有价值，有助于为整个社区改进项目。

## 贡献者

<!-- readme: collaborators,contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/uniartisan">
                    <img src="https://avatars.githubusercontent.com/u/31544054?v=4" width="100;" alt="uniartisan"/>
                    <br />
                    <sub><b>Zhiyuan Li</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/yuunnn-w">
                    <img src="https://avatars.githubusercontent.com/u/91336323?v=4" width="100;" alt="yuunnn-w"/>
                    <br />
                    <sub><b>Yuunnn_w</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/WuTianyi321">
                    <img src="https://avatars.githubusercontent.com/u/48122470?v=4" width="100;" alt="WuTianyi321"/>
                    <br />
                    <sub><b>WuTianyi</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/jiamingkong">
                    <img src="https://avatars.githubusercontent.com/u/2761215?v=4" width="100;" alt="jiamingkong"/>
                    <br />
                    <sub><b>Null</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Aknifejackzhmolong">
                    <img src="https://avatars.githubusercontent.com/u/23431216?v=4" width="100;" alt="Aknifejackzhmolong"/>
                    <br />
                    <sub><b>Dejiao Zeng</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: collaborators,contributors -end -->

****
## Technical Exchange Group

![QQ交流群](https://github.com/rwkv-kit/rwkv-kit6/blob/main/asset/qrcode_1713112204738.jpg)

**感谢各位大佬做出的贡献！欢迎各路大神为本项目提PR和Issue！你们的贡献对本项目十分有价值！！！**