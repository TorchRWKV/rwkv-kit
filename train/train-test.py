import sys
import os
# 获取当前脚本文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 'src' 目录的相对路径
src_dir = os.path.join(current_dir, '..')

# 将 'src' 目录的绝对路径添加到 Python 模块搜索路径中
sys.path.append(os.path.abspath(src_dir))

import torch
from src.model import RWKV_RNN
from src.model_utils import device_checker
from src.rwkv_tokenizer import RWKV_TOKENIZER
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import linecache

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer
        
        with open(file_path, "r") as file:
            self.total_lines = sum(1 for _ in file)

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        line = linecache.getline(self.file_path, idx + 1)
        data = json.loads(line)
        texts = data["text"]
        encoded_data = [self.tokenizer.encode(texts)[0] + [0]][0]
        
        encoded_data = torch.tensor(encoded_data, dtype=int).long()
        x = encoded_data[:-1].unsqueeze(0)
        y = encoded_data[1:].unsqueeze(0)
        return x, y
    
# 初始化模型参数
args = {
    'MODEL_NAME': './weight/ttt', #模型文件的名字，pth结尾的权重文件。
    'vocab_size': 65536 #词表大小，不要乱改
    ,'device': "cpu"
    # ,'device': "cuda"
    ,'onnx_opset':18
}
args = device_checker(args)
device = args['device']
assert device in ['cpu', 'cuda', 'musa', 'npu', 'xpu']


device = torch.device(args['device'])
# 加载模型和分词器
print("Loading model and tokenizer...")
model = RWKV_RNN(args).to(device)
tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
print("Done.")

file_path = 'data/seq.jsonl'# 替换为你的文本文件路径
save_path  = "./weight/rwkv-test-epoch-1.pth"
# 设置续写的初始字符串和参数
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
分段长度=128
dataset = TextDataset(file_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
accumulation_steps = 10 # 每 10 步更新一次参数
epochs = 1

# with torch.autograd.set_detect_anomaly(True): # 检测梯度异常
for epoch in range(epochs):
    with tqdm(dataloader) as tbar:
        for x,y in tbar:
            x=x[0]
            y=y[0]
            data_len=x.shape[1]
            state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device)
            梯度放缩比例=data_len/分段长度
            optimizer.zero_grad()
            for i in range((data_len-2)//分段长度+1):
                start=i*分段长度
                end=min((i+1)*分段长度,data_len-1)
                x_i=x[:,start:end]
                y_i=y[0,start:end]
                长度权重=x_i.shape[1]/data_len
                token_out, state_new=model.forward_parallel(x_i,state)
                state = state_new.detach()  # 使用 detach() 截断梯度传播
                loss=长度权重*criterion(token_out[0],y_i)
                # loss=loss/梯度放缩比例
                loss.backward()

            
            # loss.backward()
            if i % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            tbar.set_postfix(loss=loss.item())

model.save_model(save_path)