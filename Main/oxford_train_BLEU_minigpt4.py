import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import AutoConfig, AutoTokenizer, Gemma2ForCausalLM
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union, Any
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
# from peft import LoraConfig, TaskType, get_peft_model
from nltk.translate.bleu_score import sentence_bleu
import gc

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class OxfordDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item].cpu()
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

    def pad_emotion(self, item: int):
        emotion = self.emotion[self.image_ids[item]]
        sentiment = self.sentiment[self.image_ids[item]]
        humor = self.humor[self.image_ids[item]]
        padding_emotion = self.max_seq_len_emo - emotion.shape[0]
        padding_sentiment = self.max_seq_len_emo - sentiment.shape[0]
        padding_humor = self.max_seq_len_emo - humor.shape[0]
        if padding_emotion > 0:
            emotion = torch.cat((emotion, torch.zeros(padding_emotion, dtype=torch.int64)))
        elif padding_emotion < 0:
            emotion = emotion[:self.max_seq_len_emo]
        if padding_sentiment > 0:
            sentiment = torch.cat((sentiment, torch.zeros(padding_sentiment, dtype=torch.int64)))
        elif padding_sentiment < 0:
            sentiment = sentiment[:self.max_seq_len_emo]
        if padding_humor > 0:
            humor = torch.cat((humor, torch.zeros(padding_humor, dtype=torch.int64)))
        elif padding_humor < 0:
            humor = humor[:self.max_seq_len_emo]

        return emotion, sentiment, humor

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Any, int, Tensor, Tensor, Tensor]:
        tokens, mask = self.pad_tokens(item)
        if self.dataFrom == 'Oxford':
            prefix = torch.load('../../Oxford_HIC/ImageData/' + self.image_ids[item] + '.pt', weights_only=False)
        else:
            prefix = torch.load('../../Instagram/ImageData/' + self.dataFrom + '/' + self.image_ids[item] + '.pt',
                                weights_only=False)
        emotion, sentiment, humor = self.pad_emotion(item)
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix, item, emotion, sentiment, humor

    def filter_data_by_bleu(self, model, batch_size=20, bleu_threshold=0.1, file_path=None):
        """
        Filter dataset based on BLEU scores using batch processing.

        Args:
            model: The model used for generating sentences.
            batch_size: Number of samples per batch.
            bleu_threshold: Minimum BLEU score to keep a sample.
        """
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=False, drop_last=True)

        # Temporary storage for filtered indices
        filtered_indices = []
        device = torch.device('cuda:0')
        progress = tqdm(total=len(dataloader), desc="Filtering train dataset")
        for batch in dataloader:
            tokens_batch, masks_batch, prefixes_batch, original_indices = batch
            tokens_batch, masks_batch, prefixes_batch = tokens_batch.to(device), masks_batch.to(
                device), prefixes_batch.to(device, dtype=torch.bfloat16)

            # Forward pass
            prefix_embeds = model.clip_project(prefixes_batch).view(-1, self.prefix_length, model.embedding_size)
            text_embeds = model.falcon.model.embed_tokens(tokens_batch)
            embedding_cat = torch.cat((prefix_embeds, text_embeds), dim=1)
            outputs = model.falcon(inputs_embeds=embedding_cat, attention_mask=masks_batch)
            logits = outputs.logits[:, self.prefix_length - 1:-1]
            generated_tokens_batch = torch.argmax(logits, dim=-1)

            # BLEU calculation and filtering
            for i, original_idx in enumerate(original_indices):
                reference = self.captions_tokens[original_idx].tolist()
                candidate = generated_tokens_batch[i].tolist()
                bleu_score = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))

                if bleu_score <= bleu_threshold:
                    filtered_indices.append(original_idx)
            progress.update()

        # Update dataset based on filtered indices
        print(f"Before Filtered: {len(self.captions_tokens)}, After Filtered: {len(filtered_indices)}, BLEU <= {bleu_threshold}")
        self.captions = [self.captions[i] for i in filtered_indices]
        self.image_ids = [self.image_ids[i] for i in filtered_indices]
        self.captions_tokens = [self.captions_tokens[i] for i in filtered_indices]
        self.caption2embedding = [self.caption2embedding[i] for i in filtered_indices]
        del tokens_batch, masks_batch, prefixes_batch, original_indices, prefix_embeds, text_embeds, embedding_cat, outputs, logits, generated_tokens_batch, reference, candidate, bleu_score
        gc.collect()
        torch.cuda.empty_cache()

    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2", normalize_prefix=False, model=None,
                 batch_size=30, bleu_threshold=0.4, dataFrom='Oxford'):
        self.data_path = data_path
        self.dataFrom = dataFrom
        self.bleu = False
        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        # self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-1B-Base")
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        # self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        # self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_bleu_all.pkl"):
            del all_data
            gc.collect()
            torch.cuda.empty_cache()
            with open(f"{data_path[:-4]}_bleu_all.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
            all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
            self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(
                    torch.tensor(self.tokenizer.encode(caption['caption'], max_length=64, truncation=True),
                                 dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            self.max_seq_len = max_seq_len
            all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
            self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
            with open(f"{self.data_path[:-4]}_bleu_all.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, self.max_seq_len], f)
        if dataFrom == 'Oxford':
            self.emotions_data = pd.read_csv('../Data/Oxford_HIC/Minigpt4_Oxford.csv')
            # self.emotion_data = pd.read_csv('../Data/Oxford_HIC/Minigpt4_Oxford_emotion.csv')
            # self.sentiment_data = pd.read_csv('../Data/Oxford_HIC/Minigpt4_Oxford_sentiment.csv')
            # self.humor_data = pd.read_csv('../Data/Oxford_HIC/Minigpt4_Oxford_humor.csv')
        else:
            self.emotions_data = pd.read_csv(f'../Data/Instagram/Minigpt4_{dataFrom}.csv')
            # self.emotion_data = pd.read_csv(f'../Data/Instagram/Minigpt4_{dataFrom}_emotion.csv')
            # self.sentiment_data = pd.read_csv(f'../Data/Instagram/Minigpt4_{dataFrom}_sentiment.csv')
            # self.humor_data = pd.read_csv(f'../Data/Instagram/Minigpt4_{dataFrom}_humor.csv')
        # self.emotion will save the emotion data for each image
        self.emotion = dict()
        self.sentiment = dict()
        self.humor = dict()
        self.max_seq_len_emo = 21
        for i in range(self.emotions_data.shape[0]):
            image_id = self.emotions_data.iloc[i]['image_id']
            emotion = str(self.emotions_data.iloc[i]['emotion']).replace(';', ' ')
            sentiment = str(self.emotions_data.iloc[i]['sentiment']).replace(';', ' ')
            humor = str(self.emotions_data.iloc[i]['humor']).replace(';', ' ')
            self.emotion[image_id] = torch.tensor(self.tokenizer.encode(emotion, max_length=64, truncation=True), dtype=torch.int64)
            self.sentiment[image_id] = torch.tensor(self.tokenizer.encode(sentiment, max_length=64, truncation=True), dtype=torch.int64)
            self.humor[image_id] = torch.tensor(self.tokenizer.encode(humor, max_length=64, truncation=True, padding=True), dtype=torch.int64)

        print(f"Train Data size: {len(self.captions_tokens)}")
        # self.filter_data_by_bleu(model, batch_size, bleu_threshold)

class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item].cpu()
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
        return tokens, mask, prefix, item

    def filter_data_by_bleu(self, model, batch_size=20, bleu_threshold=0.1, file_path=None):
        """
        Filter dataset based on BLEU scores using batch processing.

        Args:
            model: The model used for generating sentences.
            batch_size: Number of samples per batch.
            bleu_threshold: Minimum BLEU score to keep a sample.
        """
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=False, drop_last=True)

        # Temporary storage for filtered indices
        filtered_indices = []
        device = torch.device('cuda:0')
        progress = tqdm(total=len(dataloader), desc="Filtering train dataset")
        for batch in dataloader:
            tokens_batch, masks_batch, prefixes_batch, original_indices = batch
            tokens_batch, masks_batch, prefixes_batch = tokens_batch.to(device), masks_batch.to(
                device), prefixes_batch.to(device, dtype=torch.bfloat16)

            # Forward pass
            # prefix_embeds = model.clip_project(prefixes_batch).view(-1, self.prefix_length, model.embedding_size)
            # text_embeds = model.falcon.model.embed_tokens(tokens_batch)
            # embedding_cat = torch.cat((prefix_embeds, text_embeds), dim=1)
            # outputs = model.falcon(inputs_embeds=embedding_cat, attention_mask=masks_batch)
            outputs = model(tokens_batch, prefixes_batch, masks_batch)
            logits = outputs.logits[:, self.prefix_length - 1: -1]
            generated_tokens_batch = torch.argmax(logits, dim=-1)

            # BLEU calculation and filtering
            for i, original_idx in enumerate(original_indices):
                reference = self.captions_tokens[original_idx].tolist()
                candidate = generated_tokens_batch[i].tolist()
                bleu_score = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))

                if bleu_score <= bleu_threshold:
                    filtered_indices.append(original_idx)
            progress.update()

        # Update dataset based on filtered indices
        print(f"Before Filtered: {len(self.captions_tokens)}, After Filtered: {len(filtered_indices)}, BLEU <= {bleu_threshold}")
        self.captions = [self.captions[i] for i in filtered_indices]
        self.image_ids = [self.image_ids[i] for i in filtered_indices]
        self.captions_tokens = [self.captions_tokens[i] for i in filtered_indices]
        self.caption2embedding = [self.caption2embedding[i] for i in filtered_indices]
        del tokens_batch, masks_batch, prefixes_batch, original_indices, outputs, logits, generated_tokens_batch, reference, candidate, bleu_score
        gc.collect()
        torch.cuda.empty_cache()

    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2", normalize_prefix=False, model=None,
                 batch_size=30, bleu_threshold=0.4):
        self.data_path = data_path
        self.bleu = False
        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        # self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-1B-Base")
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        # self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        # self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_bleu_all.pkl"):
            del all_data
            gc.collect()
            torch.cuda.empty_cache()
            with open(f"{data_path[:-4]}_bleu_all.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
            all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
            self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(
                    torch.tensor(self.tokenizer.encode(caption['caption'], max_length=64, truncation=True),
                                 dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            self.max_seq_len = max_seq_len
            all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
            self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
            with open(f"{self.data_path[:-4]}_bleu_all.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, self.max_seq_len], f)
        print(f"Train Data size: {len(self.captions_tokens)}")
        self.filter_data_by_bleu(model, batch_size, bleu_threshold)

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

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=True, dropout=0., act=nnf.relu,
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
            if (i % 2 == 0 and self.enc_dec) or (self.cross==True):  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False, cross=False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        self.cross = cross
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if (i % 2 == 0 and self.enc_dec) or (self.cross==True):
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class TransformerMapper(nn.Module):

    def forward(self, x):
        ### clip ###
        # x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        ### swin ###
        # if x.shape[2] == 768:
        #     x = self.linear(x)
        ############
        ### 768 ###
        if x.shape[2] != 768:
            x = self.linear(x)
        ############
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        ### clip ###
        # self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        ### swin ###
        # self.linear = nn.Linear(768, dim_embedding)
        ############
        ### 768 ###
        self.linear = nn.Linear(2048, dim_embedding)
        ############
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

class CrossTransformerMapper(nn.Module):

    def forward(self, x, y):
        ### clip ###
        # x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        ### swin ###
        # if x.shape[2] == 768:
        #     x = self.linear(x)
        # if y.shape[2] == 768:
        #     y = self.linear(y)
        ############
        ### 768 ###
        if x.shape[2] != 768:
            x = self.linear(x)
        if y.shape[2] != 768:
            y = self.linear(y)
        ############

        out = self.transformer(x, y)
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(CrossTransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers, cross=True)
        ### clip ###
        # self.linear = lora.Linear(dim_clip, clip_length * dim_embedding)
        ### swin ###
        # self.linear = nn.Linear(768, dim_embedding)
        ############
        ### 768 ###
        self.linear = nn.Linear(2048, dim_embedding)
        ############

class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, emotion: torch.Tensor = None, sentiment: torch.Tensor = None,
                humor: torch.Tensor = None) -> torch.Tensor:
        # embedding_text = self.gemma.model.embed_tokens(tokens)
        # embedding_text = self.gemma.base_model.model.model.embed_tokens(tokens)
        # embedding_text = self.gpt.transformer.wte(tokens)
        embedding_text = self.falcon.model.embed_tokens(tokens)
        embedding_emotion = self.falcon.model.embed_tokens(emotion)
        embedding_sentiment = self.falcon.model.embed_tokens(sentiment)
        embedding_humor = self.falcon.model.embed_tokens(humor)
        empty_ESH = torch.zeros(embedding_emotion.shape[0], 1, embedding_emotion.shape[2], dtype=torch.bfloat16,
                                device=embedding_text.device)
        embedding_ESH = torch.cat((empty_ESH, embedding_emotion, embedding_sentiment, embedding_humor), dim=1)
        # ############################################ 202502 ##############################################
        # visual_projections_swin = self.visual_project_swin(embedding_ESH, prefix)
        # visual_projections_ESH = self.visual_project_ESH(prefix, embedding_ESH)
        # ########################################################################################
        # ##### 20250226_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_add_swin_tf8 #####
        # ########################################################################################
        # # visual_projections = visual_projections_swin + visual_projections_ESH
        # ########################################################################################
        # ### 20250228_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_swin_tf8  ###
        # ########################################################################################
        # visual_projections = torch.cat((visual_projections_swin, visual_projections_ESH), dim=1)
        # visual_projections = self.visual_project(visual_projections.transpose(1, 2)).transpose(1, 2)
        # ########################################################################################
        # #######  20250304_oxford_lower_800up_only800_rest_300up_top300_ESH_co_swin_tf8   #######
        # ########################################################################################
        # # visual_projections = visual_projections_swin
        # ##################################################################################################
        # prefix_projections = self.clip_project(visual_projections).view(-1, self.prefix_length, self.embedding_size)
        # #######################################################################################
        # # 20250306_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_add_768_swin_tf8    #
        # # 20250306_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_768_swin_tf8 #
        # #######################################################################################
        # prefix_projections = self.linear(prefix_projections)
        # ################################################################################
        # embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        # ##################################################################################################

        ############################################ 202503 ##############################################
        clip_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.embedding_size)
        visual_projections_swin = self.visual_project_swin(embedding_ESH, clip_projections)
        visual_projections_ESH = self.visual_project_ESH(clip_projections, embedding_ESH)
        ########################################################################################
        ##### 20250301_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_add_swin_tf8 #####
        ########################################################################################
        # visual_projections = visual_projections_swin + visual_projections_ESH
        ########################################################################################
        ### 20250302_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_swin_tf8  ###
        ########################################################################################
        visual_projections = torch.cat((visual_projections_swin, visual_projections_ESH), dim=1)
        visual_projections = self.visual_project(visual_projections.transpose(1, 2)).transpose(1, 2)
        ########################################################################################
        #######  20250303_oxford_lower_800up_only800_rest_300up_top300_ESH_co_swin_tf8   #######
        ########################################################################################
        # visual_projections = visual_projections_swin
        ########################################################################################
        #######################################################################################
        # 20250307_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_add_768_swin_tf8    #
        # 20250307_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_768_swin_tf8 #
        #######################################################################################
        visual_projections = self.linear(visual_projections)
        ################################################################################
        embedding_cat = torch.cat((visual_projections, embedding_text), dim=1)
        ##################################################################################################

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        # out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        # out = self.gemma(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        out = self.falcon(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # self.gemma = Gemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16)
        # self.embedding_size = 2304
        # LORAconfig = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        #     inference_mode=False,  # 训练模式
        #     r=8,  # Lora 秩
        #     lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        #     lora_dropout=0.1  # Dropout 比例
        # )
        #
        # def count_trainable_parameters(model):
        #     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        #     params = sum([np.prod(p.size()) for p in model_parameters])
        #     return params
        # a = count_trainable_parameters(self.gemma)
        # self.gemma = get_peft_model(self.gemma, LORAconfig)
        # b = count_trainable_parameters(self.gemma)
        # #留下小數點後兩位就好
        # percent = round((b / a) * 100, 3)
        # print("Before: ", a, "After: ", b, "Percent: ", percent, "%")
        # self.gemma.eval()
        # for param in self.gemma.parameters():
        #     param.requires_grad = False

        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        # self.gpt = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        # self.gpt = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        # self.embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # self.gpt.eval()
        # for param in self.gpt.parameters():
        #     param.requires_grad = False

        self.falcon = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-1B-Base")
        # self.falcon = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        # self.embedding_size = self.falcon.model.embed_tokens.weight.shape[1]
        self.embedding_size = 768
        # self.falcon.eval()
        # for param in self.falcon.parameters():
        #     param.requires_grad = False
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP(
                (prefix_size, (self.embedding_size * prefix_length) // 2, self.embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.embedding_size, prefix_length, clip_length, num_layers)
        self.visual_project_swin = CrossTransformerMapper(prefix_size, self.embedding_size, prefix_length, clip_length, num_layers)
        self.visual_project_ESH = CrossTransformerMapper(prefix_size, self.embedding_size, prefix_length, clip_length, num_layers)
        #######################################################################################
        ### 20250228_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_swin_tf8 ###
        #######################################################################################
        self.visual_project = nn.Linear(128, 64)
        self.linear = nn.Linear(self.embedding_size, 2048)

class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        # self.gpt.eval()
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

def PCloss(logits: torch.Tensor, tokens: torch.Tensor, use_ce=True):
    """
    :param logits: output word logits from the generation model in shape [N, L, E]
    :param tokens:
    :param use_ce:
    :return:
    """
    EPSILON = torch.finfo(torch.bfloat16).eps
    token_length = logits.shape[1]
    vocab_size = logits.shape[2]
    loss_weight = 100
    fp_weights = [4*idx/token_length for idx in range(token_length)]
    fp_weights = torch.FloatTensor(fp_weights).to(logits.device)
    if use_ce:
        ce_thresh = 0.90
        fp_weights_ = fp_weights[None, fp_weights < ce_thresh, None]
        prob = nnf.sigmoid(logits[:, fp_weights < ce_thresh])
        label = nnf.one_hot(tokens[:, fp_weights < ce_thresh], num_classes=vocab_size)
        bce_entropy = label * torch.log(prob + EPSILON)
        bce_inv_entropy = (1 - label) * torch.log(1 - prob + EPSILON) * fp_weights_
        bce_loss = -torch.mean(bce_entropy+bce_inv_entropy)*loss_weight
        ce_loss = nnf.cross_entropy(logits[:, fp_weights >= ce_thresh].reshape(-1, logits.shape[-1]),
                                    tokens[:, fp_weights >= ce_thresh].flatten())
        loss = bce_loss + ce_loss
        return loss
    else:
        fp_weights = fp_weights[None, :, None]
        prob = nnf.sigmoid(logits)
        label = nnf.one_hot(tokens, num_classes=vocab_size)
        bce_entropy = label * torch.log(prob + EPSILON)
        bce_inv_entropy = (1 - label) * torch.log(1 - prob + EPSILON) * fp_weights
        loss = -torch.mean(bce_entropy+bce_inv_entropy)*loss_weight
        return loss

def train(trainData, testData, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "",
          bleu_batch_size=30, bleu_threshold=0.05):
    train_losses = []
    test_losses = []
    best_train_loss = 9999999999
    best_test_loss = 9999999999
    train_bleus = []
    test_bleus = []
    save = []
    bleu_threshold_list = []
    trainData_size_list = []
    testData_size_list = []

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    normalize_prefix = args.normalize_prefix
    prefix_length = args.prefix_length
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    trainDataset = OxfordDataset(trainData, prefix_length, normalize_prefix, model=model, batch_size=bleu_batch_size, bleu_threshold=bleu_threshold, dataFrom=args.dataFrom)
    testDataset = OxfordDataset(testData, prefix_length, normalize_prefix, model=model, batch_size=bleu_batch_size, bleu_threshold=bleu_threshold, dataFrom=args.dataFrom)
    train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, drop_last=True)
    epoch = 0
    while len(trainDataset) > batch_size and len(testDataset) > batch_size:
        model.train()
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
        )

        print(f">>> Training epoch {epoch + 1}")
        sys.stdout.flush()
        trainLoss = 0
        testLoss = 0
        trainBleu = 0
        testBleu = 0
        model.train()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix, original_indices, emotion, sentiment, humor) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.bfloat16)
            emotion, sentiment, humor = emotion.to(device), sentiment.to(device), humor.to(device)
            outputs = model(tokens, prefix, mask, emotion=emotion, sentiment=sentiment, humor=humor)
            logits = outputs.logits[:, trainDataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            # loss = PCloss(logits, tokens)
            trainLoss += loss.item()
            generated_tokens = torch.argmax(logits, dim=-1)
            bleu_score = sentence_bleu([tokens.flatten().tolist()], generated_tokens.flatten().tolist(),
                                       weights=(1, 0, 0, 0))
            trainBleu += bleu_score
            # loss = loss - 10 * bleu_score
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item(), "bleu": bleu_score})
            progress.update()
            # if idx % 101 == 100:
            #     break
        trainLoss /= len(train_dataloader)
        trainBleu /= len(train_dataloader)
        train_losses.append(trainLoss)
        train_bleus.append(trainBleu)
        progress.set_postfix({"loss": trainLoss, "bleu": trainBleu})
        progress.close()

        model.eval()
        with torch.no_grad():
            progress = tqdm(total=len(test_dataloader), desc=output_prefix)
            for idx, (tokens, mask, prefix, original_indices, emotion, sentiment, humor) in enumerate(test_dataloader):
                model.zero_grad()
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.bfloat16)
                emotion, sentiment, humor = emotion.to(device), sentiment.to(device), humor.to(device)

                outputs = model(tokens, prefix, mask, emotion=emotion, sentiment=sentiment, humor=humor)
                logits = outputs.logits[:, testDataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                # loss = PCloss(logits, tokens)
                testLoss += loss.item()
                generated_tokens = torch.argmax(logits, dim=-1)
                bleu_score = sentence_bleu([tokens.flatten().tolist()], generated_tokens.flatten().tolist(), weights=(1, 0, 0, 0))
                testBleu += bleu_score

                progress.set_postfix({"loss": loss.item(), "bleu": bleu_score})
                progress.update()
        testLoss /= len(test_dataloader)
        testBleu /= len(test_dataloader)
        test_losses.append(testLoss)
        test_bleus.append(testBleu)
        progress.set_postfix({"loss": testLoss, "bleu": testBleu})
        progress.close()

        if trainLoss < best_train_loss and testLoss < best_test_loss:
            best_train_loss = trainLoss
            best_test_loss = testLoss
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch + 1:03d}.pt"),
            )
            save.append('V')
        else:
            save.append(' ')

        bleu_threshold_list.append(bleu_threshold)
        trainData_size_list.append(len(trainDataset))
        testData_size_list.append(len(testDataset))

        loss_data = pd.DataFrame()
        loss_data['bleu_threshold'] = bleu_threshold_list
        loss_data['trainData_size'] = trainData_size_list
        loss_data['testData_size'] = testData_size_list
        loss_data['train_loss'] = train_losses
        loss_data['train_bleu'] = train_bleus
        loss_data['test_loss'] = test_losses
        loss_data['test_bleu'] = test_bleus
        loss_data['save'] = save
        loss_data.to_csv(f"{output_dir}/{output_prefix}-loss.csv", index=False)

        plt.plot(train_bleus, label='train_bleu')
        plt.plot(test_bleus, label='test_bleu')
        plt.legend()
        plt.savefig(f"{output_dir}/{output_prefix}-bleu.png")
        plt.show()

        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.legend()
        plt.savefig(f"{output_dir}/{output_prefix}-loss.png")
        plt.show()

        epoch += 1
        # if testBleu > bleu_threshold:
        #     bleu_threshold += 0.05
        #     model.eval()
        #     del trainDataset, testDataset, train_dataloader, test_dataloader, optimizer, scheduler, progress, tokens, mask, prefix, outputs, logits, generated_tokens, bleu_score
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     trainDataset = ClipCocoDataset(trainData, prefix_length, normalize_prefix, model=model,
        #                                    batch_size=batch_size, bleu_threshold=bleu_threshold)
        #     testDataset = ClipCocoDataset(testData, prefix_length, normalize_prefix, model=model, batch_size=batch_size,
        #                                   bleu_threshold=bleu_threshold)
        #     train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, drop_last=True)
        #     test_dataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, drop_last=True)
        #     best_train_loss = 9999999999
        #     best_test_loss = 9999999999
        #     while len(trainDataset) < batch_size or len(testDataset) < batch_size:
        #         bleu_threshold += 0.05
        #         model.eval()
        #         del trainDataset, testDataset, train_dataloader, test_dataloader
        #         gc.collect()
        #         torch.cuda.empty_cache()
        #         trainDataset = ClipCocoDataset(trainData, prefix_length, normalize_prefix, model=model,
        #                                        batch_size=batch_size, bleu_threshold=bleu_threshold)
        #         testDataset = ClipCocoDataset(testData, prefix_length, normalize_prefix, model=model,
        #                                       batch_size=batch_size, bleu_threshold=bleu_threshold)
        #         train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, drop_last=True)
        #         test_dataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, drop_last=True)
        #         best_train_loss = 9999999999
        #         best_test_loss = 9999999999
        # if bleu_threshold > 1:
        #     break
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainData', default='../Data/Oxford_HIC/parse/oxford_lower_800up_only800_all_ViT-B_32_train.pkl')
    parser.add_argument('--testData', default='../Data/Oxford_HIC/parse/oxford_lower_800up_only800_rest_300up_top300_ViT-B_32_test.pkl')
    # parser.add_argument('--trainData', default='../Data/Instagram/parse/300up_only300_all_sonicdrivein_ViT-B_32_train.pkl')
    # parser.add_argument('--testData', default='../Data/Instagram/parse/300up_only300_rest_200up_top200_sonicdrivein_ViT-B_32_test.pkl')
    parser.add_argument('--dataFrom', default='Oxford')
    parser.add_argument('--out_dir', default='20250307_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_768_swin_tf8')
    parser.add_argument('--prefix', default='checkpoint', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=64)
    parser.add_argument('--prefix_length_clip', type=int, default=64)
    parser.add_argument('--bs', type=int, default=15)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    if not os.path.exists('./Model/' + args.out_dir):
        os.makedirs('./Model/' + args.out_dir)
        os.makedirs('D:/MemeGAN/Model/' + args.out_dir)
    args.out_dir = './Model/' + args.out_dir

    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    print(args.mapping_type)
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=prefix_length, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=prefix_length, prefix_size=prefix_dim,
                                 num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    device = torch.device('cuda:0')
    model = model.to(device, dtype=torch.bfloat16)
    # 20250115_totalClip_oxford_only100_300k_transformer_p40_falcon_bleu1_0.05 == 32
    # save_file = '20250204_totalClip_oxford_1000up_only1000_rest_300up_top300_transformer_p64_falcon_swin_tf8_ins'
    # i = 27
    # model.load_state_dict(torch.load(f'./Model/{save_file}/checkpoint-{i:03d}.pt'))
    model.eval()
    train(args.trainData, args.testData, model, args, output_dir=args.out_dir, output_prefix=args.prefix
          , bleu_batch_size=20, bleu_threshold=0.05)

if __name__ == '__main__':
    main()