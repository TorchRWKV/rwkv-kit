from flask import Flask, request, Response, jsonify
from flask import stream_with_context
from torchrwkv.rwkv6 import RWKV6
import time
import uuid
import json
import argparse
from torchrwkv.model_utils import RWKVConfig

app = Flask(__name__)

# 添加跨域头信息的装饰器
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# 初始化模型和分词器

def init_model(model_path, state_path, prefill_kernel, use_jit):
    config = RWKVConfig(
        model_path=model_path,
        state_path=state_path,
        prefill_kernel=prefill_kernel,
        use_jit=use_jit
    )
    print("Loading model...")
    model = RWKV6(config=config)
    print("Done")
    return model

def format_messages_to_prompt(messages):
    formatted_prompt = ""

    # Define the roles mapping to the desired names
    role_names = {
        "system": "System",
        "assistant": "Assistant",
        "user": "User"
    }

    # Iterate through the messages and format them
    for message in messages:
        role = role_names.get(message['role'], 'Unknown')  # Get the role name, default to 'Unknown'
        content = message['content']
        formatted_prompt += f"{role}: {content}\n\n"  # Add the role and content to the prompt with newlines

    formatted_prompt += "Assistant: "
    return formatted_prompt

# 生成文本的函数
def generate_text(prompt: str, temperature=1.0, top_p=0.0, presence_penalty=0.0,
                        frequency_penalty=0.0, max_tokens=500, stop=['\n\nUser', '<|endoftext|>']):
    completion = model.generate(prompt, max_tokens, temperature, top_p)

    # 由于我们不再有tokenizer，我们无法准确计算token数量
    # 这里我们可以用字符数作为粗略估计
    prompt_tokens = len(prompt)
    completion_tokens = len(completion) - prompt_tokens
    total_tokens = prompt_tokens + completion_tokens

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }

    return completion, False, usage

# 生成文本的生成器函数
def generate_text_stream(messages, temperature=1.0, top_p=0.0, presence_penalty=0.0,
                        frequency_penalty=0.0, max_tokens=500, stop=['\n\nUser', '<|endoftext|>']):
    total_response = ""
    for chunk in model.chat(messages, max_tokens, temperature, top_p, stream=True):
        response = {
            "object": "chat.completion.chunk",
            "model": "rwkv",
            "choices": [{
                "delta": {"content": chunk},
                "index": 0,
                "finish_reason": None
            }]
        }
        total_response += chunk
        if any(stop_token in total_response for stop_token in stop):
            response["choices"][0]["finish_reason"] = "stop"
            break

        yield f"data: {json.dumps(response)}\n\n"

    # 发送结束信号
    response = {
        "object": "chat.completion.chunk",
        "model": "rwkv",
        "choices": [{
            "delta": "",
            "index": 0,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(response)}\n\n"
    yield "data: [DONE]"



# 处理 OPTIONS 请求
@app.route('/v1/chat/completions', methods=['OPTIONS'])
def options_request():
    return Response(status=200)

# 定义流式输出的路由
# Define your completion route
@app.route('/v1/chat/completions', methods=['POST'])
def create_completion():
    try:
        # Extract parameters from the request
        data = request.json
        model = data.get('model', 'rwkv')
        messages = data['messages']
        stream = data.get('stream', True)
        temperature = data.get('temperature', 1.5)
        top_p = data.get('top_p', 0.1)
        presence_penalty = data.get('presence_penalty', 0.0)
        frequency_penalty = data.get('frequency_penalty', 0.0)
        max_tokens = data.get('max_tokens', 2048)
        stop = data.get('stop', ['\n\nUser', '<|endoftext|>'])

        # Determine if streaming is enabled
        if stream:
            """
            def generate():
                for event in generate_text_stream(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop):
                    yield event
            return Response(generate(), content_type='text/event-stream')
            """
            response = Response(stream_with_context(generate_text_stream(messages, temperature=temperature, top_p=top_p, presence_penalty=presence_penalty,
                                                 frequency_penalty=frequency_penalty, max_tokens=max_tokens, stop=stop)),
                                content_type='text/event-stream')
            response.timeout = None  # 设置超时时间为无限制
            return response
        else:
            completion, if_max_token, usage = generate_text(model.apply_chat_temple(messages), temperature=temperature, top_p=top_p, presence_penalty=presence_penalty,
                                                 frequency_penalty=frequency_penalty, max_tokens=max_tokens, stop=stop)
            finish_reason = "stop" if if_max_token else "length"
            unique_id = str(uuid.uuid4())
            current_timestamp = int(time.time())
            # 检测stop标记
            for stop_token in stop:
                if stop_token in completion:
                    finish_reason = "stop"
                    completion = completion.split(stop_token)[0]
            response = {
                "id": unique_id,
                "object": "chat.completion",
                "created": current_timestamp,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion,
                    },
                    "finish_reason": finish_reason
                }],
                "usage": usage
            }
            return jsonify(response)
    except Exception as e:
        return str(e), 500

def main():
    parser = argparse.ArgumentParser(description="Run RWKV API server")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--state", type=str, default="", help="Path to the state file")
    parser.add_argument("--prefill_kernel", type=str, default="triton", help="kernel to prefill the model")
    parser.add_argument("--use_jit", type=bool, default=True, help="whether to use JIT")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8848, help="Port to run the server on")
    args = parser.parse_args()

    global model
    model = init_model(args.model, args.state, args.prefill_kernel, args.use_jit)
    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
