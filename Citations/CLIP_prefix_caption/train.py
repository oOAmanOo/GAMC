import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import json
import clip
import pickle
import argparse
import numpy as np
import pandas as pd
import loralib as lora
import matplotlib.pyplot as plt
from enum import Enum
from tqdm import tqdm
from scipy.special import softmax
from typing import Tuple, Optional, Union
from nltk.translate.bleu_score import sentence_bleu
from Citations.Parrot_Paraphraser.parrot.filters import Fluency
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class ClipCocoDataset(Dataset):

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
        image_id = self.image_ids[item]
        caption = self.captions[item]
        return tokens, mask, prefix, image_id, caption

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        ################### Load Instagram Data ###################
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        print(type(self.prefixes))
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        print("Ins Loaded")
        print(f"Prefix {self.prefixes.shape}, image_ids {len(self.image_ids)}, captions {len(self.captions)}")
        ################## Load Oxford_HIC  Data ##################
        if 'train' in data_path:
            oxford_data_path = '../../Data/Oxford_HIC/parse/oxford_3000_only1_300_8_ViT-B_32_train.pkl'
        else:
            oxford_data_path = '../../Data/Oxford_HIC/parse/oxford_3000_only1_300_2_ViT-B_32_test.pkl'
        with open(oxford_data_path, 'rb') as f:
            oxford_all_data = pickle.load(f)
        print("Data size is %0d" % len(oxford_all_data["clip_embedding"]))
        sys.stdout.flush()
        oxford_prefixes = torch.cat(oxford_all_data["clip_embedding"], dim=0)
        oxford_captions_raw = oxford_all_data["captions"]
        oxford_image_ids = [caption["image_id"] for caption in oxford_captions_raw]
        oxford_captions = [caption['caption'] for caption in oxford_captions_raw]
        print("Oxford Ins Loaded")
        print(f"Oxford Prefix {oxford_prefixes.shape}, image_ids {len(oxford_image_ids)}, captions {len(oxford_captions)}")
        self.prefixes = torch.cat((self.prefixes, oxford_prefixes), dim=0)
        self.image_ids += oxford_image_ids
        self.captions += oxford_captions
        print("ALL Loaded")
        print(f"Prefix {self.prefixes.shape}, image_ids {len(self.image_ids)}, captions {len(self.captions)}")
        ###########################################################
        self.captions_tokens = []
        self.caption2embedding = []
        max_seq_len = 0
        for caption in captions_raw:
            self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
            self.caption2embedding.append(caption["clip_embedding"])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

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
    def __init__(self, mode: str, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        if mode == 'lora':
            self.fc1 = lora.Linear(in_dim, h_dim, r=8)
            self.fc2 = lora.Linear(h_dim, out_d, r=8)
        else:
            self.fc1 = nn.Linear(in_dim, h_dim)
            self.fc2 = nn.Linear(h_dim, out_d)
        self.act = act
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, mode: str, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        if mode == 'lora':
            self.to_queries = lora.Linear(dim_self, dim_self, bias=bias, r=8)
            self.to_keys_values = lora.Linear(dim_ref, dim_self * 2, bias=bias, r=8)
            self.project = lora.Linear(dim_self, dim_self, r=8)
        else:
            self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
            self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
            self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
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

    def __init__(self, mode: str, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(mode, dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(mode, dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

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

    def __init__(self, mode: str, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(mode, dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(mode, dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(mode, dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, mode: str, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(mode, dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, mode: str, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(mode, prefix_size, self.gpt_embedding_size, prefix_length, clip_length, num_layers)

class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)

def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None, entry_length=67, temperature=1.0, stop_token: str = ".",):
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
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
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
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
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

def generate2(model, tokenizer, tokens=None, prompt=None, embed=None, entry_count=1, entry_length=67, top_p=0.8, temperature=1.0, stop_token: str = ".",):
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

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
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
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def train(train_dataset: ClipCocoDataset,test_dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    train_losses = []
    test_losses = []
    best_train_loss = 9999999999
    best_test_loss = 9999999999
    save = []

    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch+1}")
        sys.stdout.flush()
        trainLoss = 0
        testLoss = 0
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix, image_id, caption) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            trainLoss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        trainLoss /= len(train_dataloader)
        train_losses.append(trainLoss)
        progress.set_postfix({"loss": trainLoss})
        progress.close()

        model.eval()
        with torch.no_grad():
            progress = tqdm(total=len(test_data_loader))
            for idx, (tokens, mask, prefix, image_id, caption) in enumerate(test_data_loader):
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, test_dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                testLoss += loss.item()
                progress.set_postfix({"loss": testLoss})
                progress.update()
            testLoss /= len(test_data_loader)
            test_losses.append(testLoss)
            progress.set_postfix({"loss": testLoss})
            progress.close()

        if testLoss < best_test_loss and testLoss < best_test_loss:
            best_train_loss = trainLoss
            best_test_loss = testLoss
            torch.save(model.state_dict(), os.path.join(output_dir, f"{output_prefix}-{epoch + 1:03d}.pt"))
            save.append('V')
        else:
            save.append(' ')

        loss_data = pd.DataFrame()
        loss_data['train_loss'] = train_losses
        loss_data['test_loss'] = test_losses
        loss_data['save'] = save
        loss_data.to_csv(f"{output_dir}/{output_prefix}-loss.csv", index=False)

        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.legend()
        plt.savefig(f"{output_dir}/{output_prefix}-loss.png")
        plt.show()

    return model

class Predictor(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imageList = []
        self.gtList = []
        self.gt_fluency = []
        self.generate_beam_output = None
        self.generate2_output = None
        self.train_output = None
        self.generate_beam_text = []
        self.generate2_text = []
        self.train_text = []
        self.generate_beam_fitCount = []
        self.generate2_fitCount = []
        self.train_fitCount = []
        self.generate_beam_gtNum = []
        self.generate2_gtNum = []
        self.train_gtNum = []
        self.generate_beam_bleu1 = []
        self.generate_beam_bleu2 = []
        self.generate_beam_bleu3 = []
        self.generate_beam_bleu4 = []
        self.generate_beam_fluency = []
        self.generate2_bleu1 = []
        self.generate2_bleu2 = []
        self.generate2_bleu3 = []
        self.generate2_bleu4 = []
        self.generate2_fluency = []
        self.train_bleu1 = []
        self.train_bleu2 = []
        self.train_bleu3 = []
        self.train_bleu4 = []
        self.train_fluency = []
        self.train_loss = []
        self.train_caption_loss = []
        self.train_fc_loss = []
        print("parrot model loading")
        self.fluency_score  = Fluency()
        self.fluency_score.fluency_model = self.fluency_score.fluency_model.to(self.device)
        print("parrot model loaded")
        self.clip, self.clip_preprocess = clip.load("ViT-B/32")
        self.clip.to(self.device)
        self.clip.eval()
        print("clip model loaded")

    def fitCounter(self, caption, ground_truth):
        bleu1_sum = 0
        bleu2_sum = 0
        bleu3_sum = 0
        bleu4_sum = 0
        fitCount = 0
        caption_words = caption.split()
        bleu1_sum += sentence_bleu([ground_truth.split()], caption_words, weights=(1, 0, 0, 0))
        bleu2_sum += sentence_bleu([ground_truth.split()], caption_words, weights=(0, 1, 0, 0))
        bleu3_sum += sentence_bleu([ground_truth.split()], caption_words, weights=(0, 0, 1, 0))
        bleu4_sum += sentence_bleu([ground_truth.split()], caption_words, weights=(0, 0, 0, 1))
        gt_words = ground_truth.split()
        for k in range(len(caption_words)):
            if caption_words[k] in gt_words:
                fitCount += 1
        gtNum = 1

        return fitCount, gtNum, bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum

    def parrot_fluency(self, caption):
        input_ids = self.fluency_score.fluency_tokenizer("Sentence: " + caption, return_tensors='pt', truncation=True)
        input_ids = input_ids.to(self.device)
        prediction = self.fluency_score.fluency_model(**input_ids)
        scores = prediction[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        fluency_score = scores[1]  # LABEL_0 = Bad Fluency, LABEL_1 = Good Fluency
        return fluency_score

    def diversity_score(self, captionList):
        # tokenize  and extract text features
        text_tokens = clip.tokenize(captionList, context_length=77, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()
        # 計算所有 pair 的 cosine similarity
        cosine_sim = np.dot(text_features, text_features.T)  # p_m^T * p_n
        norms = np.linalg.norm(text_features, axis=1, keepdims=True)
        cosine_sim /= (norms @ norms.T)  # 除以 |pm||pn|
        # 找出每個向量最相似的另一個向量（n ≠ m）
        max_similarities = np.array([max(row[np.arange(len(row)) != i]) for i, row in enumerate(cosine_sim)])
        # 計算 diversity score
        diversity = np.sqrt(1 - np.mean(max_similarities) ** 2)
        return diversity

    def predict(self, test_dataset: ClipCocoDataset, model: ClipCaptionModel, args, epoch: int = 0):
        device = torch.device('cuda:0')
        batch_size = 1
        model = model.to(device)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        model.eval()
        with torch.no_grad():
            progress = tqdm(total=len(test_data_loader))
            for idx, (tokens, mask, prefix,  image_id, ground_truth) in enumerate(test_data_loader):
                if image_id[0] in self.imageList:
                    progress.update(1)
                    continue
                self.imageList.append(image_id[0])
                self.gtList.append(ground_truth[0])
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                prefix_embed = model.clip_project(prefix).reshape(1, test_dataset.prefix_length, -1)
                # print('--------- ground truth ---------')
                # print(ground_truth[0])
                self.gt_fluency.append(self.parrot_fluency(ground_truth[0]))
                # print('--------- generate_beam ---------')
                caption = generate_beam(model, test_dataset.tokenizer, embed=prefix_embed)[0]
                caption_token = test_dataset.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True,max_length=74)
                # print(caption)
                if self.generate_beam_output != None:
                    self.generate_beam_output = torch.cat((self.generate_beam_output, caption_token['input_ids']),dim=0)
                else:
                    self.generate_beam_output = caption_token['input_ids']
                self.generate_beam_text.append(caption)
                self.generate_beam_fluency.append(self.parrot_fluency(caption))
                fitCount, gtNum, bleu1, bleu2, bleu3, bleu4 = self.fitCounter(caption, ground_truth[0])
                self.generate_beam_fitCount.append(fitCount)
                self.generate_beam_gtNum.append(gtNum)
                self.generate_beam_bleu1.append(bleu1)
                self.generate_beam_bleu2.append(bleu2)
                self.generate_beam_bleu3.append(bleu3)
                self.generate_beam_bleu4.append(bleu4)

                # print('---------generate2---------')
                caption = generate2(model, test_dataset.tokenizer, embed=prefix_embed)
                caption_token = test_dataset.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True,max_length=74)
                # print(caption)
                if self.generate2_output != None:
                    self.generate2_output = torch.cat((self.generate2_output, caption_token['input_ids']), dim=0)
                else:
                    self.generate2_output = caption_token['input_ids']
                self.generate2_text.append(caption)
                self.generate2_fluency.append(self.parrot_fluency(caption))
                fitCount, gtNum, bleu1, bleu2, bleu3, bleu4 = self.fitCounter(caption, ground_truth[0])
                self.generate2_fitCount.append(fitCount)
                self.generate2_gtNum.append(gtNum)
                self.generate2_bleu1.append(bleu1)
                self.generate2_bleu2.append(bleu2)
                self.generate2_bleu3.append(bleu3)
                self.generate2_bleu4.append(bleu4)

                # print('---------train---------')
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, test_dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                # print(loss)
                caption_token = logits.argmax(-1)[0].cpu()
                caption = test_dataset.tokenizer.decode(caption_token, skip_special_tokens=True)
                caption_token = test_dataset.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=489)['input_ids']
                # print(caption)
                if self.train_output != None:
                    self.train_output = torch.cat((self.train_output, caption_token), dim=0)
                else:
                    self.train_output = caption_token
                self.train_text.append(caption)
                self.train_loss.append(loss.item())
                self.train_fluency.append(self.parrot_fluency(caption))
                fitCount, gtNum, bleu1, bleu2, bleu3, bleu4 = self.fitCounter(caption, ground_truth[0])
                self.train_fitCount.append(fitCount)
                self.train_gtNum.append(gtNum)
                self.train_bleu1.append(bleu1)
                self.train_bleu2.append(bleu2)
                self.train_bleu3.append(bleu3)
                self.train_bleu4.append(bleu4)
                progress.update()

            progress.close()

        generate_beam_df = pd.DataFrame()
        generate_beam_df['image_id'] = self.imageList
        generate_beam_df['caption'] = self.generate_beam_text
        generate_beam_df['fitCount'] = self.generate_beam_fitCount
        generate_beam_df['gtNum'] = self.generate_beam_gtNum
        generate_beam_df['fluency'] = self.generate_beam_fluency
        generate_beam_df['diversity'] = self.diversity_score(self.generate_beam_text)
        generate_beam_df['bleu1'] = self.generate_beam_bleu1
        generate_beam_df['bleu2'] = self.generate_beam_bleu2
        generate_beam_df['bleu3'] = self.generate_beam_bleu3
        generate_beam_df['bleu4'] = self.generate_beam_bleu4
        generate_beam_df.to_csv(f'{args.out_dir}/generate_beam_{epoch}.csv', index=False)

        generate2_df = pd.DataFrame()
        generate2_df['image_id'] = self.imageList
        generate2_df['caption'] = self.generate2_text
        generate2_df['fitCount'] = self.generate2_fitCount
        generate2_df['gtNum'] = self.generate2_gtNum
        generate2_df['fluency'] = self.generate2_fluency
        generate2_df['diversity'] = self.diversity_score(self.generate2_text)
        generate2_df['bleu1'] = self.generate2_bleu1
        generate2_df['bleu2'] = self.generate2_bleu2
        generate2_df['bleu3'] = self.generate2_bleu3
        generate2_df['bleu4'] = self.generate2_bleu4
        generate2_df.to_csv(f'{args.out_dir}/generate2_{epoch}.csv', index=False)

        train_df = pd.DataFrame()
        train_df['image_id'] = self.imageList
        train_df['caption'] = self.train_text
        train_df['fitCount'] = self.train_fitCount
        train_df['gtNum'] = self.train_gtNum
        train_df['fluency'] = self.train_fluency
        train_df['diversity'] = self.diversity_score(self.train_text)
        train_df['bleu1'] = self.train_bleu1
        train_df['bleu2'] = self.train_bleu2
        train_df['bleu3'] = self.train_bleu3
        train_df['bleu4'] = self.train_bleu4
        train_df.to_csv(f'{args.out_dir}/train_{epoch}.csv', index=False)

        ground_truth_df = pd.DataFrame()
        ground_truth_df['image_id'] = self.imageList
        ground_truth_df['caption'] = self.gtList
        ground_truth_df['fluency'] = self.gt_fluency
        ground_truth_df['diversity'] = self.diversity_score(self.gtList)
        ground_truth_df.to_csv(f'{args.out_dir}/ground_truth_{epoch}.csv', index=False)

        result = pd.DataFrame()
        result['Name'] = ['generate_beam', 'generate2', 'train', 'ground_truth']
        result['fitCount'] = [generate_beam_df['fitCount'].mean(), generate2_df['fitCount'].mean(), train_df['fitCount'].mean(), '-']
        result['gtNum'] = [generate_beam_df['gtNum'].mean(), generate2_df['gtNum'].mean(), train_df['gtNum'].mean(), '-']
        result['fluency'] = [generate_beam_df['fluency'].mean(), generate2_df['fluency'].mean(), train_df['fluency'].mean(), ground_truth_df['fluency'].mean(),]
        result['diversity'] = [generate_beam_df['diversity'].mean(), generate2_df['diversity'].mean(), train_df['diversity'].mean(), ground_truth_df['diversity'].mean()]
        result['bleu1'] = [generate_beam_df['bleu1'].mean(), generate2_df['bleu1'].mean(), train_df['bleu1'].mean(), '-']
        result['bleu2'] = [generate_beam_df['bleu2'].mean(), generate2_df['bleu2'].mean(), train_df['bleu2'].mean(), '-']
        result['bleu3'] = [generate_beam_df['bleu3'].mean(), generate2_df['bleu3'].mean(), train_df['bleu3'].mean(), '-']
        result['bleu4'] = [generate_beam_df['bleu4'].mean(), generate2_df['bleu4'].mean(), train_df['bleu4'].mean(), '-']
        result.to_csv(f'{args.out_dir}/result_{epoch}.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    ######################  mcdonalds_switzerland  ######################
    parser.add_argument('--trainData', default='../../Data/Instagram/parse/100up_only200_lessNotFunImg_53_171_passlength_12_o_mcdonalds_switzerland_ViT-B_32_train.pkl')
    parser.add_argument('--testData', default='../../Data/Instagram/parse/100up_only200_lessNotFunImg_53_171_passlength_12_x_mcdonalds_switzerland_ViT-B_32_test.pkl')
    parser.add_argument('--testData', default='../../Data/Instagram/parse/100up_only200_lessNotFunImg_53_171_passlength_12_x_mcdonalds_switzerland_ViT-B_32_testAll.pkl')
    parser.add_argument('--out_dir', default='./Model/clip_100up_only200_lessNotFunImg_53_171_passlength_12_MC_with_oxford_3000_only1_300_82')
    ##########################  sonicdrivein  ###########################
    parser.add_argument('--trainData', default='../../Data/Instagram/parse/100up_only200_lessNotFunImg_169_55_passlength_10_o_sonicdrivein_ViT-B_32_train.pkl')
    parser.add_argument('--testData', default='../../Data/Instagram/parse/100up_only200_lessNotFunImg_169_55_passlength_10_x_sonicdrivein_ViT-B_32_test.pkl')
    parser.add_argument('--testData', default='../../Data/Instagram/parse/100up_only200_lessNotFunImg_169_55_passlength_10_x_sonicdrivein_ViT-B_32_testAll.pkl')
    parser.add_argument('--out_dir', default='./Model/clip_100up_only200_lessNotFunImg_169_55_passlength_10_SD_with_oxford_3000_only1_300_82')
    #####################################################################
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=20)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    train_dataset = ClipCocoDataset(args.trainData, prefix_length, normalize_prefix=args.normalize_prefix)
    test_dataset = ClipCocoDataset(args.testData, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    model = ClipCaptionModel("lora", prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim, num_layers=args.num_layers, mapping_type=args.mapping_type)

    sys.stdout.flush()

    ####### TRAIN ########
    # train(train_dataset, test_dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)
    ####### TEST  ########
    for i in range(20):
        if os.path.exists(f'{args.out_dir}/{args.prefix}-{i + 1:03d}.pt'):
            model.load_state_dict(torch.load(f'{args.out_dir}/{args.prefix}-{i + 1:03d}.pt'))
            model = model.eval()
            pred = Predictor()
            print(f"Model {i + 1:03d} loaded.")
            pred.predict(test_dataset, model, args, i + 1)
    ######################


if __name__ == '__main__':
    main()
