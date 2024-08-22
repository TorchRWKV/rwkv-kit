# rwkv-kit

rwkv-kit is a pure PyTorch implementation of the RWKV large language model inference framework. This project aims to provide a flexible and easily scalable PyTorch implementation for the RWKV x060 model, supporting various features such as batch inference, parallel inference, ONNX format export, and standalone training.

## Features

- Native PyTorch implementation
- Batch inference support
- Parallel inference to fully leverage RWKV advantages
- Clean, readable, and easily extendable codebase
- ONNX format model export and inference
- Simple standalone training

## Hardware Support

We support various hardware devices, including but not limited to:
- NVIDIA GPUs
- Intel GPUs
- AMD GPUs
- Moore Thread MUSA GPUs
- Huawei Ascend NPUs

Contributions for additional device support are welcome.

## Installation and Usage

1. Clone the repository:
   ```
   git clone -b dev https://github.com/TorchRWKV/rwkv-kit.git
   ```

2. Install dependencies:
   ```
   cd rwkv-kit
   pip install -r requirements.txt
   ```

3. Download the RWKV6 model from [BlinkDL/rwkv-6-world](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main) and place the weights in the `weight` folder.

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
        prefill_kernel="torch", # torch, manual-torch, triton, triton-chunk
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


| Method | Batch Size | Token Length | Prefill Time (ms) | Token Generation Speed (tokens/second) | Notes |
|--------|------------|--------------|-------------------|---------------------------------------|-------|
| triton-chunk | 1 | 1024 | 132.50 | 42.83 | Suitable for inference and training, better speed |
| triton | 1 | 1024 | 105.49 | - | Suitable for inference and training, high accuracy |
| torch | 1 | 1024 | 595.22 | - | Suitable for inference on devices where Triton is unavailable |
| manual-torch | 1 | 1024 | 2468.00 | - | Suitable for training on devices where Triton is unavailable, high accuracy |
| - | 1 | - | - | 48.42 | Excluding prefill |
| - | 64 | - | - | 1266.77 | Excluding prefill |
| - | 128 | - | - | 1875.03 | Excluding prefill |

Notes:
- "-" indicates data not provided or not applicable.
- For batch size 1, only the triton-chunk method provides token generation speed (including prefill).
- For other batch sizes, token generation speeds do not include prefill time.
- Tested on WSL2, PyTorch 2.5, Intel Arc A770, 1.6B.

For normal use:
```
    initial_string = """User: 你好！ 请问你是什么模型？"""
    batch_size = 2
    state = None
    TEMPERATURE = 1.0
    TOP_P = 0.0
    LENGTH_PER_TRIAL = 100


    encoded_input = tokenizer.encode([initial_string] * batch_size)

    token = torch.tensor(encoded_input).long().to(config.device)
    token_all = torch.tensor(encoded_input).long().to(config.device)


    for step in range(LENGTH_PER_TRIAL):
        out, state = model.forward(token, state)
        token = sample_logits(out, TEMPERATURE, TOP_P)
        token_all = torch.cat((token_all, token.unsqueeze(1)), 1)

        os.system('cls' if os.name == 'nt' else 'clear')
        decoded_sequences = tokenizer.decode(token_all.cpu().tolist())
        for i, seq in enumerate(decoded_sequences):
            print(f"Batch {i+1}: {seq}")

```

You can also try:
```
    print(model.generate(initial_string, LENGTH_PER_TRIAL, TEMPERATURE, TOP_P, include_prompt=True))
    print(model.chat([{"role": "user", "content": "你是什么模型?"}], 500, TEMPERATURE, TOP_P))
    for i in model.chat([{"role": "user", "content": "你好呀"}], 500, TEMPERATURE, TOP_P, stream=True):
        print(i, end="", flush=True)
```

## ONNX Export

1. Modify parameters in `onnx_export.py` for your desired model.
2. Run:
   ```
   python onnx_export.py
   ```
3. (Optional) Create a directory for simplified models:
   ```
   mkdir ONNX_Simplified
   ```
4. (Optional) Simplify the model:
   ```
   python simplify_large_onnx.py -m onnx/{model name}.onnx -o ONNX_Simplified/{model name}.onnx
   ```
5. (Optional) Modify the model path in `onnx_infer.py` and run:
   ```
   python onnx_infer.py
   ```

To start an OpenAI server:

```
python -m rwkv-kit.openai_server --model model_path --state state_path(optional) --host 0.0.0.0 --port 8848
```

## Note

This framework currently supports only RWKV v6 models, specifically version x060.

## Future Plans

We plan to adapt this project for the AI Pro development board launched by Xunlong Orange Pi, enabling inference of the RWKV model on the Ascend ecosystem.


## Acknowledgements

Special thanks to:
- Yang, S., & Zhang, Y. (2024). FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism (Version 0.0.1) [Computer software]. https://github.com/sustcsonglin/flash-linear-attention

- [onnxsim_large_model](https://github.com/luchangli03/onnxsim_large_model.git)

We have used modified implementations based on their work in different kernels.

## Contributing

We welcome contributions from the community. Please feel free to submit PRs and raise Issues. Your input is valuable and helps improve the project for everyone.

****

优化模型用到的仓库：

## 贡献者 (Contributors)

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

****


**We warmly invite everyone to contribute to the project by submitting PRs and raising Issues! Your input and contributions are highly valued and play a vital role in improving the project for the entire community. Let's collaborate and make this project even better together!**


