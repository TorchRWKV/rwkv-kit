import linecache
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from rwkvkit.rwkv_tokenizer import RWKV_TOKENIZER
from rwkvkit.utils.rwkv6 import RWKV6
from rwkvkit.train.utils import loss_with_mask
import torch
from torch.optim.lr_scheduler import LinearLR
import torch.nn.functional as F

import linecache
from torch.utils.data import Dataset, DataLoader, Sampler
import os
import random
from typing import Dict, List
class MaskTextFolderDataset(Dataset):
    length_groups: Dict[int, List[str]]
    def __init__(self, folder_path, ctx_len, tokenizer=None, data_percentage:float=1.0):
        self.folder_path = folder_path
        self.ctx_len = ctx_len
        self.prefill_lenth = self.ctx_len + 1
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # 使用默认的tokenizer
            from rwkvkit.rwkv_tokenizer import RWKV_TOKENIZER
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            vocab_path = os.path.join(parent_dir, 'asset/rwkv_vocab_v20230424.txt')
            self.tokenizer = RWKV_TOKENIZER(vocab_path)

        self.files = self.scan_and_sort_files()
        self.length_groups = self.group_files_by_length()
        self.data_percentage = data_percentage
        self.data_index = self.create_data_index()
        print(f"Total data points: {len(self.data_index)}")  # 添加这行来打印总数据点

        self.vocab_size = 65536

    def scan_and_sort_files(self):
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.jsonl')]
        return sorted(files, key=self.extract_length_key)

    def extract_length_key(self, filename):
        parts = filename.split('-')
        if parts[-1] == 'inf.jsonl':
            return float('inf')
        return int(parts[-1].split('.')[0])

    def group_files_by_length(self):
        groups = {}
        for file in self.files:
            length = self.extract_length_key(file)
            if length not in groups:
                groups[length] = []
            groups[length].append(os.path.join(self.folder_path, file))
        return groups

    def create_data_index(self):
        data_index = []
        for length, files in self.length_groups.items():
            for file in files:
                with open(file, 'r') as f:
                    file_indices = [(file, i) for i in range(sum(1 for _ in f))]
                    # 随机选择指定百分比的数据
                    num_samples = max(1, int(len(file_indices) * self.data_percentage))
                    data_index.extend(random.sample(file_indices, num_samples))
        # random.shuffle(data_index)  # 全局打乱数据
        return data_index

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file, line_num = self.data_index[idx]
        line = linecache.getline(file, line_num + 1)
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file {file}, line {line_num + 1}")
            return self.__getitem__((idx + 1) % len(self))  # 返回下一个有效项
        texts = data["text"]


        input_output_tokens, user_assistant_tokens, user_end_positions = self.token_for_train(texts)

        input_output_tokens = torch.tensor(input_output_tokens, dtype=int).long()


        mask = self.create_mask(user_assistant_tokens)

        # check if the length of x is less than prefill_lenth
        if input_output_tokens.size(0) > self.prefill_lenth:
            input_output_tokens = input_output_tokens[:self.prefill_lenth]
            mask = mask[:self.prefill_lenth]
            # 去除大于self.prefill_lenth的数字
            user_end_positions = [i for i in user_end_positions if i < self.prefill_lenth]


        # 填充 user_end_positions 到 self.prefill_lenth
        if len(user_end_positions) == 0:
            user_end_positions = [self.prefill_lenth - 1] * 64
            mask[:] = True
        elif len(user_end_positions) < self.prefill_lenth:
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

class LengthBasedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length_groups = self.group_data_by_length()
        self.group_lengths = list(self.length_groups.keys())
        self.batches = self.create_batches()

    def group_data_by_length(self):
        groups = {}
        for idx, (file, _) in enumerate(self.dataset.data_index):
            length = self.dataset.extract_length_key(os.path.basename(file))
            if length not in groups:
                groups[length] = []
            groups[length].append(idx)
        return groups

    def create_batches(self):
        batches = []
        for length in self.group_lengths:
            group_indices = self.length_groups[length]
            random.shuffle(group_indices)

            for i in range(0, len(group_indices), self.batch_size):
                batch = group_indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def collate_fn(batch):
    # 实现填充到64的倍数的逻辑
    max_len = max(len(item[0]) for item in batch)
    padded_len = ((max_len + 63) // 64) * 64

    padded_batch = []
    for idx, targets, mask, user_end_positions in batch:
        pad_len = padded_len - len(idx)
        padded_idx = torch.cat([idx, torch.zeros(pad_len, dtype=torch.long)])
        padded_targets = torch.cat([targets, torch.zeros(pad_len, dtype=torch.long)])
        padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool)])
        padded_batch.append((padded_idx, padded_targets, padded_mask, user_end_positions))

    return torch.stack([item[0] for item in padded_batch]), \
           torch.stack([item[1] for item in padded_batch]), \
           torch.stack([item[2] for item in padded_batch]), \
           torch.stack([item[3] for item in padded_batch])



from rwkvkit.model_utils import RWKVConfig
# 初始化模型参数
config = RWKVConfig(model_path='weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
                        # state_path='weight/rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.pth',
                        prefill_kernel="triton-chunk",)
# seed = 42
torch.manual_seed(42)

# 加载模型和分词器
print("Loading model and tokenizer...")
model = RWKV6(config=config)


device = torch.device(config.device)
tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
print("Done.")

file_path = './weight/random_result_doubao_0_dataset_1.jsonl'  # 替换为你的文本文件路径
save_path = "./weight/rwkv-test-epoch-1.pth"
# 设置续写的初始字符串和参数

criterion = nn.CrossEntropyLoss()
slice_len = 1024
batch_size = 2
# 使用示例
dataset = MaskTextFolderDataset('/mnt/d/rwkv/moe/moe-test', ctx_len=512)
batch_sampler = LengthBasedBatchSampler(dataset, batch_size=2, drop_last=False)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=4)


accumulation_steps = 5  # 每 10 步更新一次参数
epochs = 6
initial_state = model.init_state(batch_size=1).detach()
initial_state.requires_grad = True
# optimizer = torch.optim.Adam([initial_state])
lr_init = 1
lr_final = 0.01
warmup_steps = 1000
total_steps =  len(dataloader) * epochs
optimizer = torch.optim.AdamW([initial_state], lr=lr_init)
scheduler = LinearLR(optimizer, start_factor=lr_init, end_factor=lr_final, total_iters=warmup_steps)


for epoch in range(epochs):
    accumulated_loss = 0
    optimizer.zero_grad()
    total_length = 0
    prev_total_length = 0
    with tqdm(dataloader) as tbar:
        for step, (idx, target, mask, user_end_list) in enumerate(tbar, start=1):
            idx = idx.to(device)
            target = target.to(device)
            mask = mask.to(device)
            data_len = idx.shape[1]
            print(data_len)
            state = initial_state.expand(batch_size, *initial_state.shape[1:]).clone()

            total_length += data_len
            prev_scale_factor = prev_total_length/total_length
            accumulated_loss *= prev_scale_factor
            loss = 0
            # 根据序列的总长度对梯度进行规范化
            param = initial_state
            if param.grad is not None:
                param.grad *= prev_scale_factor

            for i in range((data_len-2)//slice_len+1):
                start = i*slice_len
                end = min((i+1)*slice_len, data_len)
                x_i = idx[:, start:end]
                y_i = target[:, start:end]
                current_slice_len = x_i.shape[1]
                token_out, state = model.forward_prefill(x_i, state, training=True)
                loss_i = loss_with_mask(token_out, y_i, mask[:, start:end])
                print(loss_i)
                loss_weight = loss_i * (current_slice_len / total_length)
                loss += loss_weight
                accumulated_loss += loss_weight.item()

            prev_total_length = total_length

            if step % accumulation_steps == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                # 更新学习率
                if step <= warmup_steps:
                    scheduler.step()
                total_length = 0
                prev_total_length = 0
                model.save_state(initial_state, "./weight/rwkv-trained-latest.pth")
            tbar.set_postfix(avg_loss=accumulated_loss, lr=optimizer.param_groups[0]['lr'])

# 保存训练好的 state
model.save_state(initial_state, "./weight/rwkv-trained-state.pth")