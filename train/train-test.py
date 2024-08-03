import linecache
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchrwkv.rwkv_tokenizer import RWKV_TOKENIZER
from torchrwkv.rwkv6 import RWKV6
import torch
from torch.optim.lr_scheduler import LinearLR
import torch.nn.functional as F

import re
import linecache
class MaskTextDataset(Dataset):
    def __init__(self, data_file, ctx_len:int, prefill:bool, tokenizer=None, method='left'):
        self.file_path = data_file

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # 使用默认的tokenizer
            from src.rwkv_tokenizer import RWKV_TOKENIZER
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            vocab_path = os.path.join(parent_dir, 'asset/rwkv_vocab_v20230424.txt')
            self.tokenizer = RWKV_TOKENIZER(vocab_path)
        self.prefill_lenth = ctx_len + 1  # 预测下一个token，所以需要加1
        self.prefill = prefill
        self.method = method
        self.vocab_size = 65536


        with open(self.file_path, "r", encoding="utf-8") as file:
            self.total_lines = sum(1 for line in file if line.strip())

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        line = linecache.getline(self.file_path, idx + 1)
        data = json.loads(line)
        texts = data["text"]


        input_output_tokens, user_assistant_tokens, user_end_positions = self.token_for_train(texts)

        input_output_tokens = torch.tensor(input_output_tokens, dtype=int).long()


        mask = self.create_mask(user_assistant_tokens)

        # check if the length of x is less than prefill_lenth
        if self.prefill:
            if input_output_tokens.size(0) < self.prefill_lenth:
                pad_size = self.prefill_lenth - input_output_tokens.size(0)
                pad_tensor = torch.zeros((pad_size,), dtype=int).long()
                if self.method == 'left':
                    input_output_tokens = torch.cat((pad_tensor,input_output_tokens), dim=0)
                    mask = torch.cat((torch.zeros((pad_size,), dtype=bool).bool(), mask), dim=0)  # 对mask也进行填充
                    user_end_positions = [ i+pad_size for i in user_end_positions]
                else:
                    input_output_tokens = torch.cat((input_output_tokens, pad_tensor), dim=0)
                    mask = torch.cat((mask, torch.zeros((pad_size,), dtype=bool).bool()), dim=0)  # 对mask也进行填充
            elif input_output_tokens.size(0) > self.prefill_lenth:
                input_output_tokens = input_output_tokens[:self.prefill_lenth]
                mask = mask[:self.prefill_lenth]
                # 去除大于self.prefill_lenth的数字
                user_end_positions = [i for i in user_end_positions if i < self.prefill_lenth]


        # 填充 user_end_positions 到 self.prefill_lenth
        if len(user_end_positions) < self.prefill_lenth:
            user_end_positions = user_end_positions * (64 // len(user_end_positions) + 1)
            user_end_positions = user_end_positions[:64]

        idx = input_output_tokens[:-1]
        targets = input_output_tokens[1:]
        mask = mask[1:]

        user_end_positions = torch.tensor(user_end_positions, dtype=torch.int32)
        return idx, targets, mask, user_end_positions

    def create_mask(self, token_list):
        # mask 应该和y一样， encoded_data[1:]
        mask_list = []
        for i in range(len(token_list)):
            if i % 2 == 0:
                mask_list += [False] * (len(token_list[i]))
            if i % 2 == 1:  # 奇数是助手回复,所以只需要处理奇数
                mask_list += [True] * (len(token_list[i]))



        mask = torch.tensor(mask_list, dtype=bool).bool()
        return mask


    def token_for_train(self, texts):
        text_list = []  # 存储切分好的字符串
        temp_str = ""
        for i in texts:
            if i["role"] == "system":
                temp_str += "System: " + i["content"]+" \n\n"
            elif i["role"] == "user":
                temp_str += "User: " + i["content"]+" \n\nAssistant:"
                text_list.append(temp_str)
                temp_str = ""
            elif i["role"] == "assistant":
                text_list.append(" "+i["content"])


        user_assistant_tokens = []
        current_token_len = 0
        user_end_positions = []
        for i in range(len(text_list)):
            if text_list[i].startswith("System:") or text_list[i].startswith("User:"):  # 用户输入
                user_token = self.tokenizer.encode(text_list[i])[0]
                current_token_len += len(user_token)
                user_end_positions.append(current_token_len)
                user_assistant_tokens.append(user_token)
            else:  # 助手回复
                assistant_token = self.tokenizer.encode(text_list[i])[0]+[0]
                current_token_len += len(assistant_token)
                user_assistant_tokens.append(assistant_token)


        # token_list 转换为 encoder_data
        input_output_tokens = [token for text_token in user_assistant_tokens for token in text_token]
        return input_output_tokens, user_assistant_tokens, user_end_positions

from torchrwkv.model_utils import RWKVConfig
# 初始化模型参数
config = RWKVConfig(model_path='weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096',
                        state_path='weight/rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.pth',
                        prefill_kernel="manual-torch",)
# seed = 42
torch.manual_seed(42)

# 加载模型和分词器
print("Loading model and tokenizer...")
model = RWKV6(config=config)


device = torch.device(config.device)
tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
print("Done.")


# seed = 42
torch.manual_seed(42)

# 加载模型和分词器
print("Loading model and tokenizer...")
model = RWKV6(config=config)


device = torch.device(config.device)
tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
print("Done.")

file_path = 'weight/final_dataset.jsonl'  # 替换为你的文本文件路径
save_path = "./weight/rwkv-test-epoch-1.pth"
# 设置续写的初始字符串和参数

criterion = nn.CrossEntropyLoss()
slice_len = 256
batch_size = 1
dataset = MaskTextDataset(file_path, ctx_len=2048, prefill=True, tokenizer=tokenizer, method='right')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
accumulation_steps = 5  # 每 10 步更新一次参数
epochs = 1
# optimizer = torch.optim.Adam([initial_state])
lr_init = 1
lr_final = 0.01
warmup_steps = 1000
total_steps =  len(dataloader) * epochs

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def loss_with_mask(logits, target, mask):
    # 确保维度正确
    batch_size, seq_length, num_classes = logits.size()

    # 计算每个样本的有效长度
    sample_lengths = mask.float().sum(dim=1)  # [batch_size]

    # 重塑输入
    logits = logits.view(-1, num_classes)  # [batch_size * seq_length, num_classes]
    target = target.view(-1)  # [batch_size * seq_length]
    mask = mask.view(-1)  # [batch_size * seq_length]

    # 计算所有位置的交叉熵损失
    ce_loss = F.cross_entropy(logits, target, reduction='none')  # [batch_size * seq_length]

    # 应用mask
    masked_loss = ce_loss * mask  # [batch_size * seq_length]

    # 重塑回原始的batch维度
    masked_loss = masked_loss.view(batch_size, seq_length)  # [batch_size, seq_length]

    # 计算每个样本的总损失
    sample_loss = masked_loss.sum(dim=1)  # [batch_size]

    # 计算每个样本的平均损失，考虑样本长度
    avg_loss = sample_loss / (sample_lengths + 1e-8)  # 添加小值以避免除零

    # 计算整个batch的平均损失
    batch_avg_loss = avg_loss.mean()

    return batch_avg_loss


# with torch.autograd.set_detect_anomaly(True): # 检测梯度异常
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    total_tokens = 0
    optimizer.zero_grad()

    with tqdm(dataloader) as tbar:
        for step, (idx, target, mask, user_end_list) in enumerate(tbar, start=1):
            idx = idx.to(device)
            target = target.to(device)
            mask = mask.to(device)
            data_len = idx.shape[1]

            state = model.init_state(batch_size=batch_size).to(device)
            batch_loss = 0
            total_chunks = 0
            for i in range((data_len-2)//slice_len+1):
                start = i*slice_len
                end = min((i+1)*slice_len, data_len)
                x_i = idx[:, start:end]
                y_i = target[:, start:end]
                current_mask = mask[:, start:end]
                current_slice_len = x_i.shape[1]

                token_out, state = model.forward_prefill(x_i, state, training=True)
                loss_i = loss_with_mask(token_out, y_i, current_mask)

                # 计算加权loss并立即进行反向传播
                loss_weight = loss_i * (current_slice_len / data_len / accumulation_steps)
                loss_weight.backward()

                state = state.detach_()
                batch_loss += loss_i.item()
                if loss_i.item() > 0:
                    total_chunks += 1

            epoch_loss += batch_loss
            total_tokens += data_len

            if step % accumulation_steps == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            tbar.set_postfix(avg_loss=epoch_loss/total_chunks)
        torch.xpu.empty_cache()

    model.save_model(save_path)
