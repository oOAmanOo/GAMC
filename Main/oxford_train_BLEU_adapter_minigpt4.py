import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gc
import ast
import sys
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import loralib as lora
import matplotlib.pyplot as plt
from tqdm import tqdm
from enum import Enum
from torch.optim import AdamW
from typing import Tuple, Optional, Any
from peft import LoraConfig, TaskType, get_peft_model
from nltk.translate.bleu_score import sentence_bleu
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM

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

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Any, int, Tensor, Tensor, Tensor, Any]:
        tokens, mask = self.pad_tokens(item)
        if self.datafrom == 'Oxford':
            prefix = torch.load('../../Oxford_HIC/ImageData/' + self.image_ids[item] + '.pt', weights_only=False).cpu()
            emotion = torch.load('../../Oxford_HIC/ESH/bert_5384/emotion/' + self.image_ids[item] + '.pt', weights_only=False).cpu()
            sentiment = torch.load('../../Oxford_HIC/ESH/bert_5384/sentiment/' + self.image_ids[item] + '.pt', weights_only=False).cpu()
            humor = torch.load('../../Oxford_HIC/ESH/bert_5384/humor/' + self.image_ids[item] + '.pt', weights_only=False).cpu()
        else:
            prefix = torch.load('../../Instagram/ImageData/' + self.datafrom + '/' + self.image_ids[item] + '.pt', weights_only=False)
            emotion = torch.load('../../Instagram/ESH/emotion/' + self.image_ids[item] + '.pt', weights_only=False).cpu()
            sentiment = torch.load('../../Instagram/ESH/sentiment/' + self.image_ids[item] + '.pt', weights_only=False).cpu()
            humor = torch.load('../../Instagram/ESH/humor/' + self.image_ids[item] + '.pt', weights_only=False).cpu()
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        funnyscore = self.funny_scores[item]
        return tokens, mask, prefix, item, emotion, sentiment, humor, funnyscore

    def __init__(self, data_path: str, prefix_length: int, normalize_prefix=False, datafrom='Oxford'):
        self.data_path = data_path
        self.datafrom = datafrom
        self.bleu = False
        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-1B-Base")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.funny_scores = all_data["funnyscore"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]

        if os.path.isfile(f"{data_path[:-4]}_bleu_all.pkl"):
            del all_data
            gc.collect()
            torch.cuda.empty_cache()
            with open(f"{data_path[:-4]}_bleu_all.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 64
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption'], max_length=64, truncation=True),dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            self.max_seq_len = max_seq_len
            with open(f"{self.data_path[:-4]}_bleu_all.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, self.max_seq_len], f)

        if datafrom == 'Oxford':
            self.emotions_data = pd.read_csv('../Data/Oxford_HIC/Generate_original/Minigpt4_sentence_generate_Oxford/Minigpt4_sentence_generate_Oxford_bert_6844.csv')
        else:
            self.emotions_data = pd.read_csv(f'../Data/Instagram/Minigpt4_sentence_{datafrom}_bert.csv')

        self.emotion = dict()
        self.sentiment = dict()
        self.humor = dict()

        print(f"Train Data size: {len(self)}")

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, mode: str, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            if mode == 'lora':
                layers.append(lora.Linear(sizes[i], sizes[i + 1], bias=bias, r=8))
            else:
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

    def __init__(self, mode: str, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=True, dropout=0., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm):
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
            if (i % 2 == 0 and self.enc_dec) or (self.cross == True):  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, mode: str, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False,
                 cross=False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        self.cross = cross
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if (i % 2 == 0 and self.enc_dec) or (self.cross == True):
                layers.append(
                    TransformerLayer(mode, dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(mode, dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(
                    TransformerLayer(mode, dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class TransformerMapper(nn.Module):

    def forward(self, x):
        ### swin ###
        if x.shape[2] == 768:
            x = self.linear(x)
        ############
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, mode: str, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(mode, dim_embedding, 8, num_layers)
        ### swin ###
        self.linear = nn.Linear(768, dim_embedding)
        ############
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

class CrossTransformerMapper(nn.Module):

    def forward(self, x, y):
        ### swin ###
        if x.shape[2] == 768:
            x = self.linear(x)
        if y.shape[2] == 768:
            y = self.linear(y)
        ############
        out = self.transformer(x, y)
        return out

    def __init__(self, mode: str, dim_clip: int, dim_embedding: int, clip_length: int, num_layers: int = 8):
        super(CrossTransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(mode, dim_embedding, 8, num_layers, cross=True)
        ### swin ###
        self.linear = nn.Linear(768, dim_embedding)
        ############

class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, emotion: torch.Tensor = None, sentiment: torch.Tensor = None,
                humor: torch.Tensor = None) -> tuple[Any, Any] | Any:
        if self.LoRaActivated:
            embedding_text = self.falcon.base_model.model.model.embed_tokens(tokens)
        else:
            embedding_text = self.falcon.model.embed_tokens(tokens)

        ##################################################################################################
        embedding_ESH = torch.cat((emotion, sentiment, humor), dim=1)
        embedding_ESH = self.ESH_linear(embedding_ESH.transpose(1, 2)).transpose(1, 2)
        ##################################################################################################
        clip_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.embedding_size)
        visual_projections_swin = self.visual_project_swin(embedding_ESH, clip_projections)
        visual_projections_ESH = self.visual_project_ESH(clip_projections, embedding_ESH)
        ##################################################################################################
        visual_projections = torch.cat((visual_projections_swin, visual_projections_ESH), dim=1)
        visual_projections = self.visual_project(visual_projections.transpose(1, 2)).transpose(1, 2)
        ##################################################################################################
        embedding_cat = torch.cat((visual_projections, embedding_text), dim=1)
        ##################################################################################################

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.falcon(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        if self.mode == 'lora' or self.LoRaActivated:
            fc = self.funnyscore_mlp1(embedding_cat).squeeze(-1)
            fc = self.funnyscore_relu(fc)
            fc = self.funnyscore_mlp2(fc).squeeze(-1)
            fc = self.funnyscore_sigmoid(fc)
            return out, fc
        else:
            return out, 0

    def __init__(self, mode: str, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512, num_layers: int = 8):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.LoRaActivated = False
        self.mode = mode

        self.falcon = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-1B-Base")
        self.embedding_size = self.falcon.model.embed_tokens.weight.shape[1]
        self.clip_project = TransformerMapper(mode, prefix_size, self.embedding_size, prefix_length, clip_length, num_layers)
        self.visual_project_swin = CrossTransformerMapper(mode, prefix_size, self.embedding_size, clip_length, num_layers)
        self.visual_project_ESH = CrossTransformerMapper(mode, prefix_size, self.embedding_size, clip_length, num_layers)
        self.visual_project = nn.Linear(128, 64)
        self.ESH_linear = nn.Linear(64 * 3, 64)

        if mode == 'lora' or self.LoRaActivated:
            self.funnyscore_mlp1 = nn.Linear(self.embedding_size, 1)
            self.funnyscore_relu = nn.ReLU()
            self.funnyscore_mlp2 = nn.Linear(128, 1)
            self.funnyscore_sigmoid = nn.Sigmoid()

    def activateLoRa(self):
        self.LoRaActivated = True
        self.funnyscore_mlp1 = nn.Linear(self.embedding_size, 1)
        self.funnyscore_relu = nn.ReLU()
        self.funnyscore_mlp2 = nn.Linear(128, 1)
        self.funnyscore_sigmoid = nn.Sigmoid()

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

        a = count_trainable_parameters(self.falcon)
        self.falcon = get_peft_model(self.falcon, LORAconfig)
        b = count_trainable_parameters(self.falcon)
        percent = round((b / a) * 100, 3)
        print("falcon Before: ", a, "After: ", b, "Percent: ", percent, "%")

class CombinedLoss(nn.Module):

    def __init__(self, output_dir: str = ".", output_prefix: str = "", taintest:str=''):
        super(CombinedLoss, self).__init__()
        self.traintest = taintest
        self.output_dir = output_dir
        self.loss_df = pd.DataFrame(columns=["caption_loss", "fc_loss", "loss"])

    def combine_loss(self, logits: torch.Tensor, tokens: torch.Tensor, funnyscore: torch.Tensor, output_fc: torch.Tensor):
        output_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        # humor // 1 ==1 > 1 , else == 0> 0
        funnyscore = funnyscore // 1
        fc_loss = nnf.binary_cross_entropy(output_fc.flatten(), funnyscore.flatten())
        loss = output_loss + fc_loss * 10# - reward * 10
        self.loss_df = pd.concat([self.loss_df, pd.DataFrame([[output_loss.item(), fc_loss.item(), loss.item()]], columns=["caption_loss", "fc_loss", "loss"])])
        self.loss_df.to_csv(f"{self.output_dir}/{self.traintest}_separateLoss.csv", index=False)
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
    trainDataset = OxfordDataset(trainData, prefix_length, normalize_prefix, batch_size=bleu_batch_size, bleu_threshold=bleu_threshold, datafrom=args.datafrom)
    testDataset = OxfordDataset(testData, prefix_length, normalize_prefix, batch_size=bleu_batch_size, bleu_threshold=bleu_threshold, datafrom=args.datafrom)
    print(len(trainDataset), len(testDataset))
    train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    epoch = 0
    trainLoss_class = CombinedLoss(output_dir, output_prefix, 'train')
    testLoss_class = CombinedLoss(output_dir, output_prefix, 'test')

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
        for idx, (tokens, mask, prefix, original_indices, emotion, sentiment, humor, funnyscore) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.bfloat16)
            emotion, sentiment, humor = emotion.to(device, dtype=torch.bfloat16), sentiment.to(device, dtype=torch.bfloat16), humor.to(device, dtype=torch.bfloat16)
            outputs,output_fc = model(tokens, prefix, mask, emotion=emotion, sentiment=sentiment, humor=humor)
            logits = outputs.logits[:, trainDataset.prefix_length - 1: -1]
            loss = trainLoss_class.combine_loss(logits, tokens, funnyscore.to(device, dtype=torch.bfloat16), output_fc)
            trainLoss += loss.item()
            generated_tokens = torch.argmax(logits, dim=-1)
            bleu_score = sentence_bleu([tokens.flatten().tolist()], generated_tokens.flatten().tolist(),weights=(1, 0, 0, 0))
            trainBleu += bleu_score
            del tokens, mask, prefix, emotion, sentiment, humor, outputs, output_fc, logits, generated_tokens, bleu_score
            gc.collect()
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
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
            for idx, (tokens, mask, prefix, original_indices, emotion, sentiment, humor, funnyscore) in enumerate(test_dataloader):
                model.zero_grad()
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.bfloat16)
                emotion, sentiment, humor = emotion.to(device, dtype=torch.bfloat16), sentiment.to(device, dtype=torch.bfloat16), humor.to(device, dtype=torch.bfloat16)
                outputs, output_fc = model(tokens, prefix, mask, emotion=emotion, sentiment=sentiment, humor=humor)
                logits = outputs.logits[:, trainDataset.prefix_length - 1: -1]
                loss = testLoss_class.combine_loss(logits, tokens, funnyscore.to(device, dtype=torch.bfloat16), output_fc)
                testLoss += loss.item()
                generated_tokens = torch.argmax(logits, dim=-1)
                bleu_score = sentence_bleu([tokens.flatten().tolist()], generated_tokens.flatten().tolist(), weights=(1, 0, 0, 0))
                testBleu += bleu_score
                del tokens, mask, prefix, emotion, sentiment, humor, outputs, output_fc, logits, generated_tokens
                gc.collect()
                torch.cuda.empty_cache()
                progress.set_postfix({"loss": loss.item()})
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
    return model

def main():
    parser = argparse.ArgumentParser()
    ########################  pre-trained  model ########################
    # parser.add_argument('--trainData', default='../Data/Oxford_HIC/parse/oxford_3000_only1_300_8_ViT-B_32_train.pkl')
    # parser.add_argument('--testData', default='../Data/Oxford_HIC/parse/oxford_3000_only1_300_2_ViT-B_32_test.pkl')
    ######################  mcdonalds_switzerland  ######################
    parser.add_argument('--trainData', default='../Data/Instagram/parse/100up_only200_lessNotFunImg_53_171_passlength_12_o_mcdonalds_switzerland_ViT-B_32_train.pkl')
    parser.add_argument('--testData', default='../Data/Instagram/parse/100up_only200_lessNotFunImg_53_171_passlength_12_x_mcdonalds_switzerland_ViT-B_32_test.pkl')
    parser.add_argument('--datafrom', default='mcdonalds_switzerland')
    parser.add_argument('--out_dir', default='20250514_100up_only200_lessNotFunImg_53_171_passlength_12_MC_base_0421_oxford_3000_only1_300_82_ESH_filter_cross_concat_combineLoss')
    ##########################  sonicdrivein  ###########################
    # parser.add_argument('--trainData', default='../Data/Instagram/parse/100up_only200_lessNotFunImg_169_55_passlength_10_o_sonicdrivein_ViT-B_32_train.pkl')
    # parser.add_argument('--testData', default='../Data/Instagram/parse/100up_only200_lessNotFunImg_169_55_passlength_10_x_sonicdrivein_ViT-B_32_test.pkl')
    # parser.add_argument('--datafrom', default='sonicdrivein')
    # parser.add_argument('--out_dir', default='20250428_100up_only200_lessNotFunImg_169_55_passlength_10_SD_base_0421_oxford_3000_only1_300_82_ESH_filter_cross_concat_combineLoss')
    #####################################################################
    parser.add_argument('--prefix', default='checkpoint', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=64)
    parser.add_argument('--prefix_length_clip', type=int, default=64)
    parser.add_argument('--bs', type=int, default=20)
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
    device = torch.device('cuda:0')
    def count_trainable_parameters(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    origin_model = ClipCaptionModel(mode='nn', prefix_length=prefix_length, clip_length=prefix_length, prefix_size=prefix_dim, num_layers=args.num_layers)
    model = ClipCaptionModel(mode='lora', prefix_length=prefix_length, clip_length=prefix_length, prefix_size=prefix_dim, num_layers=args.num_layers)
    print(count_trainable_parameters(origin_model))

    ####################  Load the pre-trained model ####################
    save_file = '20250421_oxford_3000_only1_300_82_ESH_bert_cross_concat'
    i = 4
    origin_model.load_state_dict(torch.load(f'./Model/{save_file}/checkpoint-{i:03d}.pt'))
    print(count_trainable_parameters(origin_model))
    #####################################################################

    #######################   Activate adapters   #######################
    def weightToLora(fullModel, newModel):
        for name, param in fullModel.named_parameters():
            newModel.state_dict()[name].copy_(param.data)

        for name, param in newModel.named_parameters():
            if 'falcon' not in name:
                # if 'clip_project.' in name or 'visual_project_swin.' in name or 'visual_project_ESH.' in name:
                # if 'clip_project.' not in name:
                if "lora_" not in name:  # 只讓 LoRA 參數訓練
                    if 'bias' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                    if 'funnyscore' in name:
                        param.requires_grad = True
                else:
                    param.requires_grad = True
                # else:
                #     param.requires_grad = True

        for name, param in newModel.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        return newModel

    a = count_trainable_parameters(origin_model)
    model = weightToLora(origin_model, model)
    del origin_model
    gc.collect()
    torch.cuda.empty_cache()
    b = count_trainable_parameters(model)
    percent = round((b / a) * 100, 3)
    print("Before: ", a, "After: ", b, "Percent: ", percent, "%")
    #####################################################################

    ####################### Activate LoRA in LLM  #######################
    # model.activateLoRa()
    # b = count_trainable_parameters(model)
    # percent = round((b / a) * 100, 3)
    # print("Before: ", a, "After: ", b, "Percent: ", percent, "%")
    #####################################################################

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    train(args.trainData, args.testData, model, args, output_dir=args.out_dir, output_prefix=args.prefix, bleu_batch_size=20, bleu_threshold=0.05)

if __name__ == '__main__':
    main()