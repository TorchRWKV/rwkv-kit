import torch
import torch.nn.functional as F
import re
import json
import linecache
from torch.utils.data import Dataset


class MaskTextDataset(Dataset):
    def __init__(
            self,
            data_file,
            ctx_len: int,
            prefill: bool,
            tokenizer=None,
            method='left'):
        self.file_path = data_file

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # 使用默认的tokenizer
            from rwkvkit.rwkv_tokenizer import RWKV_TOKENIZER
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            vocab_path = os.path.join(
                parent_dir, 'asset/rwkv_vocab_v20230424.txt')
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

        input_output_tokens, user_assistant_tokens, user_end_positions = self.token_for_train(
            texts)

        input_output_tokens = torch.tensor(
            input_output_tokens, dtype=int).long()

        mask = self.create_mask(user_assistant_tokens)

        # check if the length of x is less than prefill_lenth
        if self.prefill:
            if input_output_tokens.size(0) < self.prefill_lenth:
                pad_size = self.prefill_lenth - input_output_tokens.size(0)
                pad_tensor = torch.zeros((pad_size,), dtype=int).long()
                if self.method == 'left':
                    input_output_tokens = torch.cat(
                        (pad_tensor, input_output_tokens), dim=0)
                    mask = torch.cat(
                        (torch.zeros((pad_size,), dtype=bool).bool(), mask), dim=0)  # 对mask也进行填充
                    user_end_positions = [
                        i + pad_size for i in user_end_positions]
                else:
                    input_output_tokens = torch.cat(
                        (input_output_tokens, pad_tensor), dim=0)
                    mask = torch.cat(
                        (mask, torch.zeros(
                            (pad_size,), dtype=bool).bool()), dim=0)  # 对mask也进行填充
            elif input_output_tokens.size(0) > self.prefill_lenth:
                input_output_tokens = input_output_tokens[:self.prefill_lenth]
                mask = mask[:self.prefill_lenth]
                # 去除大于self.prefill_lenth的数字
                user_end_positions = [
                    i for i in user_end_positions if i < self.prefill_lenth]

        # 填充 user_end_positions 到 self.prefill_lenth
        if len(user_end_positions) < self.prefill_lenth:
            user_end_positions = user_end_positions * \
                (64 // len(user_end_positions) + 1)
            user_end_positions = user_end_positions[:64]

        idx = input_output_tokens[:-1]
        targets = input_output_tokens[1:]
        mask = mask[1:]

        user_end_positions = torch.tensor(
            user_end_positions, dtype=torch.int32)
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
        texts_list_split = re.split(
            r"(?=\n\nUser: |\n\nAssistant: )", texts)  # 使用正则表达式切分文本

        for i in range(len(texts_list_split)):
            if i % 2 == 0:  # 用户输入
                text_list.append(texts_list_split[i])
            else:  # 助手回复
                text_list.append(
                    texts_list_split[i].replace(
                        '\n\nAssistant: ',
                        '').replace(
                        "\n\n<|endoftext|>",
                        '').replace(
                        "<|endoftext|>\n\n",
                        "").replace(
                        "<|endoftext|>",
                        ""))

        user_assistant_tokens = []
        current_token_len = 0
        user_end_positions = []
        for i in range(len(texts_list_split)):
            if i % 2 == 0:  # 用户输入
                # + self.tokenizer.encode("\n\nAssistant: ")[0]
                user_token = self.tokenizer.encode(
                    text_list[i] + "\n\nAssistant: ")[0]
                current_token_len += len(user_token)
                user_end_positions.append(current_token_len)
                user_assistant_tokens.append(user_token)
            else:  # 助手回复
                assistant_token = self.tokenizer.encode(text_list[i])[0] + [0]
                current_token_len += len(assistant_token)
                user_assistant_tokens.append(assistant_token)

        # token_list 转换为 encoder_data
        input_output_tokens = [
            token for text_token in user_assistant_tokens for token in text_token]
        return input_output_tokens, user_assistant_tokens, user_end_positions


def loss_with_mask(logits, target, response_mask):
    # logits.shape = [bs, seq_len, vocab_size],  (seq_len=max_len - 1),    target.shape = [bs, seq_len]
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=0, reduce=False)  # loss.shape = [bs * seq_len]
    loss_mask = response_mask.view(-1) # [bs * seqlen]
    loss = torch.sum(loss*loss_mask) / (loss_mask.sum() + 1e-8)
    return loss