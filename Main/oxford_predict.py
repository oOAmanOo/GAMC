# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md
from torch.utils.data import Dataset, DataLoader
import clip
import os
import pandas as pd
import pickle
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers import AutoConfig, AutoTokenizer, Gemma2ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import PIL.Image

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

WEIGHTS_PATHS = {
    "coco": "coco_weights.pt",
    "conceptual-captions": "conceptual_weights.pt",
}

D = torch.device
CPU = torch.device("cpu")


class Predictor(object):
    def __init__(self, prefix_length, cp_num):
        self.generate_beam_output = None
        self.generate2_output = None
        self.train_output = None
        self.generate_beam_text = []
        self.generate2_text = []
        self.train_text = []
        self.train_loss = []
        self.prefix_length = prefix_length
        # self.embedding_size = 768
        self.embedding_size = 2304
        self.cp_num = cp_num

    def predict(self, tokens, masks, prefixs, text_list, model):
        """Run a single prediction on the model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens, masks, prefixs = tokens.to(device), masks.to(device), prefixs.to(device, dtype=torch.bfloat16)
        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        model.eval()
        model.to(device, dtype=torch.bfloat16)
        if len(text_list) == 2:
            data_mode = "test"
        else:
            data_mode = "train"
        for i, text in enumerate(text_list):
            print('====================== '+data_mode+' '+ str(i+1) +'======================')
            print('--------- ground truth ---------')
            print(text)
            print('--------- generate_beam ---------')
            prefix_embed = model.clip_project(prefixs[i].unsqueeze(0)).view(-1, self.prefix_length, self.embedding_size).to(device, dtype=torch.bfloat16)
            caption = generate_beam(model, tokenizer, embed=prefix_embed)[0]
            caption_token = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=74)
            print(caption)
            if self.generate_beam_output != None:
                self.generate_beam_output = torch.cat((self.generate_beam_output, caption_token['input_ids']), dim=0)
            else:
                self.generate_beam_output = caption_token['input_ids']
            self.generate_beam_text.append(caption)
            print('---------generate2---------')
            caption = generate2(model, tokenizer, embed=prefix_embed)
            caption_token = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=74)
            print(caption)
            if self.generate2_output != None:
                self.generate2_output = torch.cat((self.generate2_output, caption_token['input_ids']), dim=0)
            else:
                self.generate2_output = caption_token['input_ids']
            self.generate2_text.append(caption)
            print('---------train---------')
            # embedding_text = model.gemma.model.embed_tokens(tokens[i].unsqueeze(0))
            embedding_text = model.gemma.base_model.model.model.embed_tokens(tokens[i].unsqueeze(0))
            # embedding_text = model.gpt.transformer.wte(tokens[i].unsqueeze(0))
            print(prefix_embed.shape, embedding_text.shape)
            embedding_cat = torch.cat((prefix_embed, embedding_text), dim=1)
            out = model.gemma(inputs_embeds=embedding_cat, attention_mask=masks[i].unsqueeze(0))
            # out = model.gpt(inputs_embeds=embedding_cat, attention_mask=masks[i].unsqueeze(0))
            logits = out.logits[:, self.prefix_length-1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[i].flatten(), ignore_index=0)
            print(loss)
            caption_token = logits.argmax(-1)[0].cpu()
            caption = tokenizer.decode(caption_token, skip_special_tokens=True)
            caption_token = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=489)['input_ids']
            print(caption)
            if self.train_output != None:
                self.train_output = torch.cat((self.train_output, caption_token), dim=0)
            else:
                self.train_output = caption_token
            self.train_text.append(caption)
            self.train_loss.append(loss.item())

        if len(text_list) != 2:
            def dataframe_Name(name, count=489, rows=1):
                if rows == 1:
                    name_df = pd.DataFrame([[name]], columns=["Name"])
                    num_df = pd.DataFrame([[0] * count for _ in range(rows)], columns=[f"{i}" for i in range(0, count)])
                    name_df = pd.concat([name_df, num_df], axis=1)
                    return name_df
                else:
                    name_df = pd.DataFrame([[name] * count for _ in range(rows)],columns=[f"{i}" for i in range(489 - count, 489)])
                    return name_df

            test = pd.DataFrame()
            test['Name'] = ['test1', 'test2', 'train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7',
                            'train8', 'train9',
                            'train10']
            generate_beam_df = pd.DataFrame(self.generate_beam_output.cpu().detach(),columns=[f"{i}" for i in range(0, 74)])
            generate_beam_df = pd.concat([test, generate_beam_df, dataframe_Name("-", 74, 12)], axis=1)
            generate_beam_df['text'] = self.generate_beam_text
            generate2_df = pd.DataFrame(self.generate2_output.cpu().detach(), columns=[f"{i}" for i in range(0, 74)])
            generate2_df = pd.concat([test, generate2_df, dataframe_Name("-", 74, 12)], axis=1)
            generate2_df['text'] = self.generate2_text
            train_df = pd.DataFrame(self.train_output.cpu().detach(), columns=[f"{i}" for i in range(0, 489)])
            train_df = pd.concat([test, train_df], axis=1)
            train_df['loss'] = self.train_loss
            train_df['text'] = self.train_text
            print(dataframe_Name("generate_beam").shape, generate_beam_df.shape, dataframe_Name("generate2").shape,
                  generate2_df.shape, dataframe_Name("train").shape, train_df.shape)
            final = pd.concat([dataframe_Name("generate_beam"), generate_beam_df, dataframe_Name("generate2"),
                               generate2_df, dataframe_Name("train"), train_df], axis=0)
            print(final.shape)
            print(final.columns)

            final.to_csv(f'./Model/{save_file}/{save_file}_test_{self.cp_num:03d}.csv', index=False)




class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention

class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gemma = Gemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto",
                                                       torch_dtype=torch.bfloat16)
        self.embedding_size = 2304
        LORAconfig = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,  # 训练模式
            r=8,  # Lora 秩
            lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1  # Dropout 比例
        )

        def count_trainable_parameters(model):
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            return params

        a = count_trainable_parameters(self.gemma)
        self.gemma = get_peft_model(self.gemma, LORAconfig)
        b = count_trainable_parameters(self.gemma)
        # 留下小數點後兩位就好
        percent = round((b / a) * 100, 3)
        print("Before: ", a, "After: ", b, "Percent: ", percent, "%")
        # self.gemma.eval()
        # for param in self.gemma.parameters():
        #     param.requires_grad = False

        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        # self.gpt.eval()
        # for param in self.gpt.parameters():
        #     param.requires_grad = False
        # self.embedding_size = self.gpt.transformer.wte.weight.shape[1]

        # if mapping_type == MappingType.MLP:
        # self.clip_project = MLP((prefix_size, (self.embedding_size * prefix_length) // 2,
        #                              self.embedding_size * prefix_length))
        # else:
        self.clip_project = TransformerMapper(prefix_size, self.embedding_size, prefix_length,
                                                                     clip_length, num_layers)

class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None, entry_length=74, temperature=1.0, stop_token: str = ".", ):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                # generated = model.gpt.transformer.wte(tokens)
                # generated = model.gemma.model.embed_tokens(tokens)
                generated = model.gemma.base_model.model.model.embed_tokens(tokens)
        for i in range(entry_length):
            outputs = model.gemma(inputs_embeds=generated)
            # outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            # next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            # next_token_embed = model.gemma.model.embed_tokens(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            next_token_embed = model.gemma.base_model.model.model.embed_tokens(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

def generate2(model, tokenizer, tokens=None, prompt=None, embed=None, entry_count=1, entry_length=74, top_p=0.8, temperature=1.0, stop_token: str = ".", ):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                # generated = model.gpt.transformer.wte(tokens)
                # generated = model.gemma.model.embed_tokens(tokens)
                generated = model.gemma.base_model.model.model.embed_tokens(tokens)
            for i in range(entry_length):
                outputs = model.gemma(inputs_embeds=generated)
                # outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                # next_token_embed = model.gemma.model.embed_tokens(next_token)
                next_token_embed = model.gemma.base_model.model.model.embed_tokens(next_token)
                # next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break
            if tokens.shape == torch.Size([1, 1]):
                output_list = [tokens.squeeze().item()]  # 如果是标量或 1-d，则获取值作为列表
            else:
                 output_list = list(tokens.squeeze().cpu().numpy())  # 按正常方式处理
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

class TrainClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

class TestClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainData = '../Data/Oxford_HIC/parse/oxford_100k_ViT-B_32_train.pkl'
testData = '../Data/Oxford_HIC/parse/oxford_100k_ViT-B_32_test.pkl'
prefix_length = 10
normalize_prefix = True
trainDataset = TrainClipCocoDataset(trainData, prefix_length, normalize_prefix=normalize_prefix)
testDataset = TestClipCocoDataset(testData, prefix_length, normalize_prefix=normalize_prefix)
# oxford_300k
# train_image = ['imgflip_34', 'bokete_3820', 'imgflip_0','imgflip_8', 'imgflip_15', 'imgflip_19', 'bokete_104530','imgflip_730', 'imgflip_130', 'imgflip_677']
# train_text = ['You finish doing something at your friends house and look at your phone; 7 missed calls from your mom; 7 missed calls from your mom'
#               ,'I\'m in my 50s!'
#               ,'School; Memes'
#               ,'image tagged in memes,one does not simply'
#               ,'THAT; IS WHAT A GOOD MEME LOOKS LIKE'
#               ,'NOT SURE IF PEOPLE ARE UPVOTING MEMES; OR USER NAMES'
#               ,'It\'s a family night runaway.'
#               ,'CHUCK IS THE GOOD TYPE OF SCUMBAG; CUZ HE ONLY ROASTS YOU FROM YOUR INSIDES'
#               ,'SO YOUR TELLIN\' ME THAT SCHOOLS GOOD FOR YOU'
#               ,'Y\'ALL GOT ANY MORE OF THEM; JOBS?']
# 100K
train_image = ['imgflip_34', 'bokete_3820', 'imgflip_0','imgflip_8', 'imgflip_15', 'imgflip_19', 'bokete_104530','imgflip_730', 'imgflip_130', 'imgflip_677']
train_text = ['You finish doing something at your friends house and look at your phone; 7 missed calls from your mom; 7 missed calls from your mom'
              ,'I\'m in my 50s!'
              ,'School; Memes'
              ,'image tagged in memes,one does not simply'
              ,'THAT; IS WHAT A GOOD MEME LOOKS LIKE'
              ,'NOT SURE IF PEOPLE ARE UPVOTING MEMES; OR USER NAMES'
              ,'It\'s a family night runaway.'
              ,'Chuck Norris doesn\'t go washroom; He goes washBOOM!'
              ,'SO YOUR TELLIN\' ME THAT SCHOOLS GOOD FOR YOU'
              ,'Y\'ALL GOT ANY MORE OF THEM; JOBS?']
tokens_list = []
mask_list = []
prefix_list = []
train_caption =[]
image_id_list = []
for i in range(len(trainDataset)):
    caption = trainDataset.captions[i]
    image_id = trainDataset.image_ids[i]
    if image_id in train_image:
        if caption in train_text and image_id not in image_id_list:
            tokens, mask, prefix = trainDataset[i]
            tokens_list.append(tokens)
            mask_list.append(mask)
            prefix_list.append(prefix)
            train_caption.append(caption)
            image_id_list.append(image_id)
train_tokens = torch.stack(tokens_list).to(device)
train_mask = torch.stack(mask_list).to(device)
train_prefix = torch.stack(prefix_list).to(device)
print(train_tokens.shape, train_mask.shape, train_prefix.shape)
# # 300K
# # Image ID: imgflip_7, Caption: CHEESE; ME AT 3 AM; CHEESE; MY MOM WHO WAS WAITING; ME
# # Image ID: imgflip_32, Caption: IS THIS A PIGEON?
# test_image = ['imgflip_7', 'imgflip_32']
# test_text = ['CHEESE; ME AT 3 AM; CHEESE; MY MOM WHO WAS WAITING; ME'
#               ,'IS THIS A PIGEON?']
# # 100K
test_image = ['imgflip_7', 'imgflip_32']
test_text = ['THE DOG FOOD; MY DOG; DOG FOOD; ME LOOKING AT HIM; MY DOG'
              ,'MATH; ME; IS THIS THE REASON I DROPPED OUT OF COLLEGE?']
# Image ID: bokete_111723, Caption: I'm going to take care of you. I'm going to take care of you. I'm going to take care of you.
# Image ID: imgflip_57, Caption: WHAT'S DONE IN THE DARK WILL ALWAYS COME OUT IN THE LIGHT; BUT THATS NONE OF MY BUSINESS
# test_image = ['bokete_111723', 'imgflip_57']
# test_text = ['I\'m going to take care of you. I\'m going to take care of you. I\'m going to take care of you.'
#               ,'WHAT\'S DONE IN THE DARK WILL ALWAYS COME OUT IN THE LIGHT; BUT THATS NONE OF MY BUSINESS']
tokens_list = []
mask_list = []
prefix_list = []
test_caption =[]
image_id_list = []
for i in range(len(testDataset)):
    caption = testDataset.captions[i]
    image_id = testDataset.image_ids[i]
    if image_id in test_image:
        if caption in test_text and image_id not in image_id_list:
            # print(f"Image ID: {image_id}, Caption: {caption}")
            tokens, mask, prefix = testDataset[i]
            tokens_list.append(tokens)
            mask_list.append(mask)
            prefix_list.append(prefix)
            test_caption.append(caption)
            image_id_list.append(image_id)
test_tokens = torch.stack(tokens_list).to(device)
test_mask = torch.stack(mask_list).to(device)
test_prefix = torch.stack(prefix_list).to(device)
print(test_tokens.shape, test_mask.shape, test_prefix.shape)


model = ClipCaptionModel( prefix_length, clip_length=prefix_length, prefix_size=512, num_layers=8)
save_file = '20241231_totalClip_oxford_100K_transformer_p10_lora'
for i in range(16):
    if os.path.exists(f'./Model/{save_file}/checkpoint-{i+1:03d}.pt'):
        model.load_state_dict(torch.load(f'./Model/{save_file}/checkpoint-{i+1:03d}.pt'))
        model = model.eval()
        model = model.to(device, dtype=torch.bfloat16)
        pred = Predictor(prefix_length, cp_num = i+1)
        pred.predict(test_tokens, test_mask, test_prefix, test_caption, model)
        pred.predict(train_tokens, train_mask, train_prefix, train_caption, model)

AllCaption =pd.DataFrame()
for i in range(16):
    if os.path.exists(f'./Model/{save_file}/checkpoint-{i+1:03d}.pt'):
        df = pd.read_csv(f'./Model/{save_file}/{save_file}_test_{i+1:03d}.csv')
        textAndLoss = df[['text', 'loss']]
        AllCaption = pd.concat([AllCaption, textAndLoss], axis=1)
AllCaption.to_csv(f'./Model/{save_file}/{save_file}_test_all.csv', index=False)

