import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gc
import sys
import clip
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Optional, Union, Any
from nltk.translate.bleu_score import sentence_bleu
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn import functional as nnf
from torch.nn.functional import normalize
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss
import torchmetrics
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model
from transformers import get_linear_schedule_with_warmup

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
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        ### swin ###
        # if x.shape[2] == 768:
        #     x = self.linear(x)
        ############
        ### 768 ###
        # if x.shape[2] != 768:
        #     x = self.linear(x)
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
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        ### swin ###
        # self.linear = nn.Linear(768, dim_embedding)
        ############
        ### 768 ###
        # self.linear = nn.Linear(2048, dim_embedding)
        ############
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

class CrossTransformerMapper(nn.Module):

    def forward(self, x, y):
        ### clip ###
        # x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        ### swin ###
        if x.shape[2] == 768:
            x = self.linear(x)
        if y.shape[2] == 768:
            y = self.linear(y)
        ############
        ### 768 ###
        # if x.shape[2] != 768:
        #     x = self.linear(x)
        # if y.shape[2] != 768:
        #     y = self.linear(y)
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
        self.linear = nn.Linear(768, dim_embedding)
        ############
        ### 768 ###
        # self.linear = nn.Linear(2048, dim_embedding)
        ############

class ImageTextModel(nn.Module):
    def __init__(self, gpt2_model_name="gpt2", feature_dim=768, output_dim=1):
        super(ImageTextModel, self).__init__()

        # Image Encoder (ResNet-50)
        self.resnet = models.resnet50(pretrained=True)
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) #torch.Size([10, 2048, 1, 1])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) #torch.Size([10, 2048, 7, 7])
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, feature_dim)  # Modify FC layer
        # self.fusion = nn.Linear(feature_dim, 128)
        self.resnet_linear = nn.Linear(2048, feature_dim)
        ### 倒數第二層

        # self.clip_project = TransformerMapper(2048, 768, 64, 64, num_layers=8)

        # Text Encoder (GPT-2)
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        # self.text_proj = nn.Linear(self.gpt2.config.hidden_size, feature_dim)

        # Fusion Layer
        self.transformer = Transformer(768, 8, 8)
        # contrastive learning
        self.temp = nn.Parameter(0.4 * torch.ones(1), requires_grad=True)
        # self.fusion = nn.Linear(feature_dim * 2, 128)

        ######## mlp ########
        self.mlp1 = nn.Linear((49 + 64) * 768, 1024)
        self.mlp2 = nn.Linear(1024, 64)
        ######## mlp 3 layers 0414 ########
        # self.mlp1 = nn.Linear((49+64)*768, 2048)
        # self.mlp15 = nn.Linear(2048, 1024)
        # self.mlp2 = nn.Linear(1024, 64)
        ######## mlp 3 layers 0415 ########
        # self.mlp1 = nn.Linear((49 + 64) * 768, 8192)
        # self.mlp15 = nn.Linear(8192, 1024)
        # self.mlp2 = nn.Linear(1024, 64)
        ######## mlp 3 layers 0414 ########

        self.classifier = nn.Linear(64, output_dim)
        ### 98k > 1k > 64 > 1
        # self.classifier = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification

    def forward(self, image, input_ids, attention_mask):
        # Encode image
        img_features = self.resnet(image).squeeze(-1).squeeze(-1)  # Shape: (batch_size, 2048)
        # prefix_projections = self.clip_project(img_features).view(-1, 64, 768)
        img_features = img_features.view(img_features.shape[0], 2048, -1) # Shape: (batch_size, 2048, 7*7)
        img_features = self.resnet_linear(img_features.transpose(1, 2))  # Shape: (batch_size, 7*7, 768)
        # Encode text (Get last hidden state, take CLS token representation)
        text_outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        # text_features = self.text_proj(text_features)

        # Fusion (Concatenation + Projection)
        fused = torch.cat((img_features, text_outputs), dim=1)
        fused = self.transformer(fused)
        mix_features = fused.view(fused.shape[0], -1)
        # contrastive learning
        sim_features = normalize(mix_features, dim=-1)
        sim = sim_features @ sim_features.T
        sim = sim / self.temp

        # humor_fused

        # Classification
        # fused = torch.cat((img_features, text_features), dim=1)
        # fused = img_features
        logits = self.mlp1(mix_features)
        logits = self.mlp2(logits)
        logits = self.classifier(logits)
        probs = self.sigmoid(logits).squeeze(-1)  # Shape: (batch_size, output_dim)

        return probs, sim

class Dataset(torch.utils.data.Dataset):
    def get_image_features(self, img_id, humor):
        if self.traintest != 'test'and self.traintest != 'train':
            file = f"../../humorscore_image_{self.traintest}_data/{img_id}.pt"
            if os.path.exists(file):
                return torch.load(file)
            else:
                filename = f"../Data/Instagram/{self.traintest}_img/{img_id}.jpg"
                image = Image.open(filename).convert('RGB')
                image = self.image_transform(image)
                torch.save(image, f"../../humorscore_image_{self.traintest}_data/{img_id}.pt")
                return image
        elif os.path.exists(f"../../humorscore_image_{self.traintest}_data/oxford_{img_id}.pt"):
            return torch.load(f"../../humorscore_image_{self.traintest}_data/oxford_{img_id}.pt")
        elif os.path.exists(f"../../humorscore_image_{self.traintest}_data/COCO_{int(img_id):012d}.pt"):
            return torch.load(f"../../humorscore_image_{self.traintest}_data/COCO_{int(img_id):012d}.pt")


    def get_caption_embedding(self, item):
        inputs = self.tokenizer([str(self.caption_list[item])], truncation=True, max_length=64, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        padding = torch.zeros(1, 64 - input_ids.shape[1], dtype=torch.int64)
        input_ids = torch.cat((input_ids, padding), dim=1).squeeze(0)
        attention_mask = torch.cat((attention_mask, padding), dim=1).squeeze(0)
        return input_ids, attention_mask

    def __getitem__(self, item: int):
        image = self.get_image_features(self.image_list[item], self.humor[item])
        caption_id, caption_attmask = self.get_caption_embedding(item)
        humor = self.humor[item]
        rank = self.rank[item]
        return image, caption_id, caption_attmask, rank
        # return image, caption_id, caption_attmask, humor, rank, str(self.image_list[item]), str(self.caption_list[item])

    def __len__(self):
        return len(self.image_list)

    def __init__(self, oxford_data: pd.DataFrame, traintest: str, dataPath: str):
        device = torch.device('cuda:0')
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 doesn’t have a pad token by default
        self.traintest = traintest
        # Define image transform
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_list = []
        self.caption_list = []
        self.humor = []
        self.rank = []
        if self.traintest == 'train':
            out_path = f"../Data/{dataPath}_train.pkl"
            self.coco_dir = 'C:/Users/user/fiftyone/coco-2014/raw/captions_train2014.json'
            self.coco_image_dir = 'C:/Users/user/fiftyone/coco-2014/train/data/COCO_train2014_'
        elif self.traintest == 'test':
            out_path = f"../Data/{dataPath}_test.pkl"
            self.coco_dir = 'C:/Users/user/fiftyone/coco-2014/raw/captions_val2014.json'
            self.coco_image_dir = 'C:/Users/user/fiftyone/coco-2014/validation/data/COCO_val2014_'
        else:
            out_path = f"../Data/{dataPath}_{self.traintest}.pkl"


        if os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                alldata = pickle.load(f)
            self.image_list = alldata['image_list']
            self.caption_list = alldata['caption_list']
            self.humor = alldata['humor']
            self.rank = alldata['rank']
            print('Data Loaded')
            print("%0d embeddings saved " % len(self.image_list))
        else:
            # if self.traintest == 'train' or self.traintest == 'test':
                # with open(self.coco_dir, 'r') as f:
                #     data = json.load(f)
                # data = data['annotations']
                # print("%0d captions loaded from json " % len(data))
                #
                # with tqdm(total=len(data)) as pbar:
                #     for i in range(len(data)):
                #         if traintest == 'test' and i > 414113/4:
                #             break
                #         d = data[i]
                #         self.image_list.append(d["image_id"])
                #         self.caption_list.append(d['caption'])
                #         self.humor.append(torch.tensor([0]))
                #         self.rank.append(torch.tensor([0]))
                #         if (i + 1) % 10000 == 0:
                #             with open(out_path, 'wb') as f:
                #                 pickle.dump({"image_list": self.image_list,
                #                              "caption_list": self.caption_list,
                #                              "humor": torch.cat(self.humor, dim=0),
                #                              "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
                #             pbar.set_postfix({"present": i})
                #         pbar.update(1)
                # with open(out_path, 'wb') as f:
                #     pickle.dump({"image_list": self.image_list,
                #                  "caption_list": self.caption_list,
                #                  "humor": torch.cat(self.humor, dim=0),
                #                  "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
                # print('COCO Done')
                # print("%0d embeddings saved " % len(self.image_list))
                # pbar.close()
            if self.traintest == 'train' or self.traintest == 'test':
                std = oxford_data['funny_score_y'].std().item()
                mean = oxford_data['funny_score_y'].mean().item()
                with tqdm(total=len(oxford_data)) as pbar:
                    for i in range(len(oxford_data)):
                        d = oxford_data.iloc[i]
                        d = d.to_dict()
                        self.image_list.append(d["image_id"])
                        self.caption_list.append(d['caption'])
                        self.humor.append(torch.tensor([1]))
                        columnName = 'funny_score_y'
                        if d[columnName] > (mean + 2 * std):
                            self.rank.append(torch.tensor([1.0]))
                        elif d[columnName] > (mean + 1 * std):
                            self.rank.append(torch.tensor([1.0]))
                            # self.rank.append(torch.tensor([0.75]))
                        elif d[columnName] > (mean):
                            self.rank.append(torch.tensor([1.0]))
                            # self.rank.append(torch.tensor([0.5]))
                        elif d[columnName] > (mean - 1 * std):
                            self.rank.append(torch.tensor([0.0]))
                            # self.rank.append(torch.tensor([0.25]))
                        else:
                            self.rank.append(torch.tensor([0.0]))

                        if (i + 1) % 10000 == 0:
                            with open(out_path, 'wb') as f:
                                pickle.dump({"image_list": self.image_list,
                                             "caption_list": self.caption_list,
                                             "humor": torch.cat(self.humor, dim=0),
                                             "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
                            pbar.set_postfix({"present": i})
                        pbar.update(1)
            else:
                with tqdm(total=len(oxford_data)) as pbar:
                    for i in range(len(oxford_data)):
                        d = oxford_data.iloc[i]
                        d = d.to_dict()
                        self.image_list.append(d["image_id"])
                        self.caption_list.append(d['caption'])
                        self.humor.append(torch.tensor([0]))
                        if d['funny_score'] >= 0.9:
                            self.rank.append(torch.tensor([1.0]))
                        elif d['funny_score'] > 0.5:
                            self.rank.append(torch.tensor([1.0]))
                            # self.rank.append(torch.tensor([0.5]))
                        else:
                            self.rank.append(torch.tensor([0.0]))

                        if (i + 1) % 10000 == 0:
                            with open(out_path, 'wb') as f:
                                pickle.dump({"image_list": self.image_list,
                                             "caption_list": self.caption_list,
                                             "humor": torch.cat(self.humor, dim=0),
                                             "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
                            pbar.set_postfix({"present": i})
                        pbar.update(1)

            with open(out_path, 'wb') as f:
                pickle.dump({"image_list": self.image_list,
                             "caption_list": self.caption_list,
                             "humor": torch.cat(self.humor, dim=0),
                             "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
            print('Oxford/ins Done')
            print("%0d embeddings saved " % len(self.image_list))
            pbar.close()

class MixDataset(torch.utils.data.Dataset):
    def get_image_features(self, img_id, humor):
        if os.path.exists(f"../../humorscore_image_{self.traintest}_data/oxford_{img_id}.pt"):
            return torch.load(f"../../humorscore_image_{self.traintest}_data/oxford_{img_id}.pt")
        if str(img_id).isnumeric():
            if os.path.exists(f"../../humorscore_image_{self.traintest}_data/COCO_{int(img_id):012d}.pt"):
                return torch.load(f"../../humorscore_image_{self.traintest}_data/COCO_{int(img_id):012d}.pt")
        if os.path.exists(f"../../humorscore_image_sonicdrivein_data/{img_id}.pt"):
            return torch.load(f"../../humorscore_image_sonicdrivein_data/{img_id}.pt")
        if os.path.exists(f"../../humorscore_image_mcdonalds_switzerland_data/{img_id}.pt"):
            return torch.load(f"../../humorscore_image_mcdonalds_switzerland_data/{img_id}.pt")
        else:
            filename = f"../Data/Oxford_HIC/oxford_img/{img_id}.jpg"
            image = Image.open(filename).convert('RGB')
            image = self.image_transform(image)
            torch.save(image, f"../../humorscore_image_{self.traintest}_data/oxford_{img_id}.pt")
            return image
    def get_caption_embedding(self, item):
        inputs = self.tokenizer([str(self.caption_list[item])], truncation=True, max_length=64, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        padding = torch.zeros(1, 64 - input_ids.shape[1], dtype=torch.int64)
        input_ids = torch.cat((input_ids, padding), dim=1).squeeze(0)
        attention_mask = torch.cat((attention_mask, padding), dim=1).squeeze(0)
        return input_ids, attention_mask

    def __getitem__(self, item: int):
        # rank vs humor(0, 1)
        image = self.get_image_features(self.image_list[item], self.humor[item])
        caption_id, caption_attmask = self.get_caption_embedding(item)
        humor = self.humor[item]
        rank = self.rank[item]

        # return image, caption_id, caption_attmask, rank
        return image, caption_id, caption_attmask, humor, rank, str(self.image_list[item]), str(self.caption_list[item])

    def __len__(self):
        return len(self.image_list)

    def __init__(self, oxford_data: pd.DataFrame, ins_data: pd.DataFrame, traintest: str, dataPath: str):
        device = torch.device('cuda:0')
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 doesn’t have a pad token by default
        self.traintest = traintest
        # Define image transform
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_list = []
        self.caption_list = []
        self.humor = []
        self.rank = []
        if self.traintest == 'train':
            out_path = f"../Data/{dataPath}_train.pkl"
            self.coco_dir = 'C:/Users/user/fiftyone/coco-2014/raw/captions_train2014.json'
            self.coco_image_dir = 'C:/Users/user/fiftyone/coco-2014/train/data/COCO_train2014_'
        elif self.traintest == 'test':
            out_path = f"../Data/{dataPath}_test.pkl"
            self.coco_dir = 'C:/Users/user/fiftyone/coco-2014/raw/captions_val2014.json'
            self.coco_image_dir = 'C:/Users/user/fiftyone/coco-2014/validation/data/COCO_val2014_'
        else:
            out_path = f"../Data/{dataPath}_{self.traintest}.pkl"


        if os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                alldata = pickle.load(f)
            self.image_list = alldata['image_list']
            self.caption_list = alldata['caption_list']
            self.humor = alldata['humor']
            self.rank = alldata['rank']
            print('Data Loaded')
            print("%0d embeddings saved " % len(self.image_list))
        else:

            # with open(self.coco_dir, 'r') as f:
            #     data = json.load(f)
            # data = data['annotations']
            # print("%0d captions loaded from json " % len(data))
            #
            # with tqdm(total=len(data)) as pbar:
            #     for i in range(len(data)):
            #         if traintest == 'test' and i > 414113/4:
            #             break
            #         d = data[i]
            #         self.image_list.append(d["image_id"])
            #         self.caption_list.append(d['caption'])
            #         self.humor.append(torch.tensor([0]))
            #         self.rank.append(torch.tensor([0]))
            #         if (i + 1) % 10000 == 0:
            #             with open(out_path, 'wb') as f:
            #                 pickle.dump({"image_list": self.image_list,
            #                              "caption_list": self.caption_list,
            #                              "humor": torch.cat(self.humor, dim=0),
            #                              "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
            #             pbar.set_postfix({"present": i})
            #         pbar.update(1)
            # with open(out_path, 'wb') as f:
            #     pickle.dump({"image_list": self.image_list,
            #                  "caption_list": self.caption_list,
            #                  "humor": torch.cat(self.humor, dim=0),
            #                  "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
            # print('COCO Done')
            # print("%0d embeddings saved " % len(self.image_list))
            # pbar.close()

            std = oxford_data['funny_score_y'].std().item()
            mean = oxford_data['funny_score_y'].mean().item()
            with tqdm(total=len(oxford_data)) as pbar:
                for i in range(len(oxford_data)):
                    d = oxford_data.iloc[i]
                    d = d.to_dict()
                    self.image_list.append(d["image_id"])
                    self.caption_list.append(d['caption'])
                    self.humor.append(torch.tensor([1]))
                    if d['funny_score_y'] > 0.5:
                        self.rank.append(torch.tensor([1.0]))
                    else:
                        self.rank.append(torch.tensor([0.0]))
                    # if d['funny_score_y'] > (mean + 2 * std):
                    #     self.rank.append(torch.tensor([1.0]))
                    # elif d['funny_score_y'] > (mean + 1 * std):
                    #     self.rank.append(torch.tensor([0.75]))
                    # elif d['funny_score_y'] > (mean):
                    #     self.rank.append(torch.tensor([0.5]))
                    # elif d['funny_score_y'] > (mean - 1 * std):
                    #     self.rank.append(torch.tensor([0.25]))
                    # else:
                    #     self.rank.append(torch.tensor([0.0]))

                    if (i + 1) % 10000 == 0:
                        with open(out_path, 'wb') as f:
                            pickle.dump({"image_list": self.image_list,
                                         "caption_list": self.caption_list,
                                         "humor": torch.cat(self.humor, dim=0),
                                         "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
                        pbar.set_postfix({"present": i})
                    pbar.update(1)
            with open(out_path, 'wb') as f:
                pickle.dump({"image_list": self.image_list,
                             "caption_list": self.caption_list,
                             "humor": torch.cat(self.humor, dim=0),
                             "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
            print('Oxford Done')
            print("%0d embeddings saved " % len(self.image_list))
            pbar.close()

            with tqdm(total=len(ins_data)) as pbar:
                for i in range(len(ins_data)):
                    d = ins_data.iloc[i]
                    d = d.to_dict()
                    self.image_list.append(d["image_id"])
                    self.caption_list.append(d['caption'])
                    self.humor.append(torch.tensor([0]))
                    if d['funny_score'] >= 0.9:
                        self.rank.append(torch.tensor([1.0]))
                    elif d['funny_score'] >= 0.5:
                        self.rank.append(torch.tensor([0.5]))
                    else:
                        self.rank.append(torch.tensor([0.0]))

                    if (i + 1) % 10000 == 0:
                        with open(out_path, 'wb') as f:
                            pickle.dump({"image_list": self.image_list,
                                         "caption_list": self.caption_list,
                                         "humor": torch.cat(self.humor, dim=0),
                                         "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
                        pbar.set_postfix({"present": i})
                    pbar.update(1)
            with open(out_path, 'wb') as f:
                pickle.dump({"image_list": self.image_list,
                             "caption_list": self.caption_list,
                             "humor": torch.cat(self.humor, dim=0),
                             "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
            print('ins Done')
            print("%0d embeddings saved " % len(self.image_list))
            pbar.close()

class InsDataset(torch.utils.data.Dataset):
    def get_image_features(self, img_id, humor):
        if os.path.exists(f"../../humorscore_image_sonicdrivein_data/{img_id}.pt"):
            return torch.load(f"../../humorscore_image_sonicdrivein_data/{img_id}.pt")
        if os.path.exists(f"../../humorscore_image_mcdonalds_switzerland_data/{img_id}.pt"):
            return torch.load(f"../../humorscore_image_mcdonalds_switzerland_data/{img_id}.pt")
        # file = f"../../humorscore_image_{self.traintest}_data/{img_id}.pt"
        # if os.path.exists(file):
        #     return torch.load(file)
        # else:
        #     filename = f"../Data/Instagram/{self.traintest}_img/{img_id}.jpg"
        #     image = Image.open(filename).convert('RGB')
        #     image = self.image_transform(image)
        #     torch.save(image, f"../../humorscore_image_{self.traintest}_data/{img_id}.pt")
        #     return image

    def get_caption_embedding(self, item):
        inputs = self.tokenizer([str(self.caption_list[item])], truncation=True, max_length=64, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        padding = torch.zeros(1, 64 - input_ids.shape[1], dtype=torch.int64)
        input_ids = torch.cat((input_ids, padding), dim=1).squeeze(0)
        attention_mask = torch.cat((attention_mask, padding), dim=1).squeeze(0)
        return input_ids, attention_mask

    def __getitem__(self, item: int):
        image = self.get_image_features(self.image_list[item], self.humor[item])
        caption_id, caption_attmask = self.get_caption_embedding(item)
        humor = self.humor[item]
        rank = self.rank[item]
        # return image, caption_id, caption_attmask, rank
        return image, caption_id, caption_attmask, humor, rank, str(self.image_list[item]), str(self.caption_list[item])

    def __len__(self):
        return len(self.image_list)

    def __init__(self, ins_data: pd.DataFrame, traintest: str, dataPath: str):
        device = torch.device('cuda:0')
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 doesn’t have a pad token by default
        self.traintest = traintest
        # Define image transform
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_list = []
        self.caption_list = []
        self.humor = []
        self.rank = []
        if self.traintest == 'train':
            out_path = f"../Data/{dataPath}_train.pkl"
        elif self.traintest == 'test':
            out_path = f"../Data/{dataPath}_test.pkl"
        else:
            out_path = f"../Data/{dataPath}_{self.traintest}.pkl"

        if os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                alldata = pickle.load(f)
            self.image_list = alldata['image_list']
            self.caption_list = alldata['caption_list']
            self.humor = alldata['humor']
            self.rank = alldata['rank']
            print(self.rank)
            print('Data Loaded')
            print("%0d embeddings saved " % len(self.image_list))
        else:
            with tqdm(total=len(ins_data)) as pbar:
                for i in range(len(ins_data)):
                    d = ins_data.iloc[i]
                    d = d.to_dict()
                    self.image_list.append(d["image_id"])
                    self.caption_list.append(d['caption'])
                    self.humor.append(torch.tensor([0]))
                    if d['funny_score'] >= 0.9:
                        self.rank.append(torch.tensor([1.0]))
                    elif d['funny_score'] >= 0.5:
                        self.rank.append(torch.tensor([0.5]))
                    else:
                        self.rank.append(torch.tensor([0.0]))

                    if (i + 1) % 10000 == 0:
                        with open(out_path, 'wb') as f:
                            pickle.dump({"image_list": self.image_list,
                                         "caption_list": self.caption_list,
                                         "humor": torch.cat(self.humor, dim=0),
                                         "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
                        pbar.set_postfix({"present": i})
                    pbar.update(1)

            with open(out_path, 'wb') as f:
                pickle.dump({"image_list": self.image_list,
                             "caption_list": self.caption_list,
                             "humor": torch.cat(self.humor, dim=0),
                             "rank": torch.cat(self.rank, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
            print('ins Done')
            print("%0d embeddings saved " % len(self.image_list))
            pbar.close()

class FocalContrastiveLoss(nn.Module):
    def __init__(self, output_dir: str = ".", output_prefix: str = "", taintest:str=''):
        super(FocalContrastiveLoss, self).__init__()
        self.traintest = taintest
        self.output_dir = output_dir
        self.loss_df = pd.DataFrame(columns=["focalLoss", "contrastive_loss", "loss"])

    def computeLoss(self, pred, target, sim):
        alpha = 0.9
        gamma = 1.6
        # Focal Loss
        focalLoss = sigmoid_focal_loss(pred, target, alpha=alpha, gamma=gamma, reduction='mean')

        # Contrastive Loss
        contrast_target = (target.unsqueeze(0) == target.unsqueeze(1)).float()
        mask = torch.eye(sim.shape[0], device=sim.device).bool()
        contrast_target = contrast_target.masked_fill(mask, 0)
        sim = sim.masked_fill(mask, -1e9)
        contrastive_loss = nn.BCEWithLogitsLoss()(sim, contrast_target)

        # Combine losses
        loss = focalLoss * 10 + contrastive_loss
        if self.traintest == 'test' or self.traintest == 'train':
            self.loss_df = pd.concat([self.loss_df, pd.DataFrame([[focalLoss.item(), contrastive_loss.item(), loss.item()]], columns=["focalLoss", "contrastive_loss", "loss"])])
            self.loss_df.to_csv(f"./Model/{self.output_dir}/{self.traintest}_separateLoss.csv", index=False)

        return loss

def train(model, args, output_dir: str = ".", output_prefix: str = ""):
    train_losses = []
    test_losses = []
    best_train_loss = 9999999999
    best_test_loss = 9999999999
    save = []

    # former = pd.read_csv(f"'./Model/'{output_dir}/{output_prefix}-loss.csv")
    # train_losses = list(former['train_loss'])
    # test_losses = list(former['test_loss'])
    # save = list(former['save'])
    # best_train_loss = min(train_losses)
    # best_test_loss = min(test_losses)

    device = torch.device('cuda:0')
    batch_size = args.bs
    model = model.to(device)
    dataPath = args.dataPath

    if os.path.exists(f"../Data/{dataPath}_train.pkl") and os.path.exists(f"../Data/{dataPath}_test.pkl"):
        if os.path.exists(f"../Data/{dataPath}_train.pkl"):
            # trainDataset = Dataset(pd.DataFrame(), 'train', dataPath)
            # trainDataset = InsDataset(pd.DataFrame(), 'train', dataPath)
            trainDataset = MixDataset(pd.DataFrame(), pd.DataFrame(), 'train', dataPath)
        if os.path.exists(f"../Data/{dataPath}_test.pkl"):
            # testDataset = Dataset(pd.DataFrame(), 'test', dataPath)
            # testDataset = InsDataset(pd.DataFrame(), 'test', dataPath)
            testDataset = MixDataset(pd.DataFrame(), pd.DataFrame(), 'test', dataPath)
    else:
        # ##################   oxford   ##################
        # data = pd.read_csv('../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv')
        # threshold = data['funny_score_y'].quantile(0.75)
        # data = data[data['funny_score_y'] >= threshold]
        # unique_image_ids = data['image_id'].unique()
        # train_ids, test_ids = train_test_split(unique_image_ids, test_size=0.2, random_state=42)
        # oxford_train = data[data['image_id'].isin(train_ids)]
        # oxford_test = data[data['image_id'].isin(test_ids)]
        # print(f'oxford: {oxford_train.shape}, {oxford_test.shape}')
        ######################## only 1 1:1 ins
        data = pd.read_csv('../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv')
        fcmean = data['funny_score_y'].mean()
        data = (
            data.sort_values(by=['image_id', 'funny_score_y'], ascending=[True, False])
            .groupby('image_id')
            .head(1)
        )
        data = data[:2100]
        data['funny_score_y'] = data['funny_score_y'].apply(lambda x: 1 if x > fcmean else 0)
        oxford_train, oxford_test = train_test_split(data, test_size=0.2, random_state=42)
        print(f'oxford: {oxford_train.shape}, {oxford_test.shape}')
        #################   instagram   #################
        data = pd.read_csv('../Data/Instagram/Generate_sonicdrivein.csv')
        # print("shape of data: ", data.shape)
        image_id_counts = data['image_id'].value_counts()
        threshold = 10

        original = pd.read_csv('../Data/Instagram/Filter_sonicdrivein.csv')
        original['caption'] = original['caption'].str.lower()
        data['caption'] = data['caption'].str.lower()
        original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
        original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)
        original = original[original['gen_count'] >= 100]
        original_train = original[original['text_len'] >= threshold]
        ins_train = original_train
        # original_train = original[:1136]
        # train = data.merge(original_train, on='image_id', how='inner', suffixes=('', '_'))
        # ins_train = (
        #     train.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
        #     .groupby('image_id')
        #     .head(200)
        # )
        original_test = original[original['text_len'] < threshold]
        # original_test = original[~original['image_id'].isin(ins_train['image_id'])]
        ins_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:(len(ins_train['image_id'].unique()) // 4)]
        print(f'sonic_train: {ins_train.shape}, sonic_test: {ins_test.shape}')
        data = pd.read_csv('../Data/Instagram/Generate_mcdonalds_switzerland.csv')
        # print("shape of data: ", data.shape)
        image_id_counts = data['image_id'].value_counts()
        threshold = 12

        original = pd.read_csv('../Data/Instagram/Filter_mcdonalds_switzerland.csv')
        original['caption'] = original['caption'].str.lower()
        data['caption'] = data['caption'].str.lower()
        original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
        original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)
        original = original[original['gen_count'] >= 100]
        original_train = original[original['text_len'] >= threshold]
        temp_train = original_train
        # original_train = original[:621]
        # train = data.merge(original_train, on='image_id', how='inner', suffixes=('', '_'))
        # temp_train = (
        #     train.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
        #     .groupby('image_id')
        #     .head(200)
        # )
        ins_train = pd.concat([ins_train, temp_train], ignore_index=True)
        original_test = original[original['text_len'] < threshold]
        # original_test = original[~original['image_id'].isin(ins_train['image_id'])]
        temp_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:(len(temp_train['image_id'].unique()) // 4)]
        ins_test = pd.concat([ins_test, temp_test], ignore_index=True)
        print(f'mcd_train: {temp_train.shape}, mcd_test: {temp_test.shape}')
        print(f'instagram: {ins_train.shape}, {ins_test.shape}')

        ins_train['funny_score'] = ins_train['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        ins_test['funny_score'] = ins_test['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        ################################################
        # trainDataset = Dataset(oxford_train, 'train', dataPath)
        # testDataset = Dataset(oxford_test, 'test', dataPath)
        # trainDataset = InsDataset(ins_train, 'train', dataPath)
        # testDataset = InsDataset(ins_test, 'test', dataPath)
        trainDataset = MixDataset(oxford_train, ins_train, 'train', dataPath)
        testDataset = MixDataset(oxford_test, ins_test, 'test', dataPath)
    # get data size
    print(len(trainDataset), len(testDataset))
    train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    trainLoss_class = FocalContrastiveLoss(output_dir, output_prefix, 'train')
    testLoss_class = FocalContrastiveLoss(output_dir, output_prefix, 'test')
    epoch = 0
    while len(trainDataset) > batch_size and len(testDataset) > batch_size:
        optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
        # criterion = nn.BCELoss()
        # criterion = nn.MSELoss()
        print(f">>> Training epoch {epoch + 1}")
        sys.stdout.flush()
        trainLoss = 0
        testLoss = 0
        model.train()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (images, caption_ids, caption_masks, humor, rank, img_id, caption) in enumerate(train_dataloader):
            model.zero_grad()
            images, caption_ids, caption_masks, rank = images.to(device), caption_ids.to(device), caption_masks.to(device), rank.to(device)
            outputs, sim = model(images, caption_ids, caption_masks)
            loss = trainLoss_class.computeLoss(outputs, rank, sim)
            trainLoss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            # if idx % 101 == 100:
            #     break
        trainLoss /= len(train_dataloader)
        train_losses.append(trainLoss)
        progress.set_postfix({"loss": trainLoss})
        progress.close()

        model.eval()
        with torch.no_grad():
            progress = tqdm(total=len(test_dataloader), desc=output_prefix)
            for idx, (images, caption_ids, caption_masks, humor, rank, img_id, caption) in enumerate(test_dataloader):
                model.zero_grad()
                images, caption_ids, caption_masks, rank = images.to(device), caption_ids.to(device), caption_masks.to(device), rank.to(device)
                outputs, sim = model(images, caption_ids, caption_masks)
                loss = testLoss_class.computeLoss(outputs, rank, sim)
                testLoss += loss.item()
                progress.set_postfix({"loss": loss.item()})
                progress.update()
        testLoss /= len(test_dataloader)
        test_losses.append(testLoss)
        progress.set_postfix({"loss": testLoss})
        progress.close()

        if trainLoss < best_train_loss and testLoss < best_test_loss:
            best_train_loss = trainLoss
            best_test_loss = testLoss
            torch.save(
                model.state_dict(),
                f"./Model/{output_dir}/checkpoint-{epoch + 1:03d}.pt"
            )
            save.append('V')
        else:
            save.append(' ')

        loss_data = pd.DataFrame()
        loss_data['train_loss'] = train_losses
        loss_data['test_loss'] = test_losses
        loss_data['save'] = save
        loss_data.to_csv(f"./Model/{output_dir}/{output_prefix}-loss.csv", index=False)

        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.legend()
        plt.savefig(f"./Model/{output_dir}/{output_prefix}-loss.png")
        plt.show()

        epoch += 1
    return model

def test(model, args, output_dir: str = ".", output_prefix: str = ""):
    indomain_losses = []
    mcdonald_losses = []
    sonic_losses = []
    indomain_accuracy = []
    mcdonald_accuracy = []
    sonic_accuracy = []
    indomain_accuracy_rank = []
    mcdonald_accuracy_rank = []
    sonic_accuracy_rank = []
    indomain_tp = []
    mcdonald_tp = []
    sonic_tp = []
    indomain_tn = []
    mcdonald_tn = []
    sonic_tn = []
    indomain_fp = []
    mcdonald_fp = []
    sonic_fp = []
    indomain_fn = []
    mcdonald_fn = []
    sonic_fn = []
    indomain_mae = []
    mcdonald_mae = []
    sonic_mae = []
    indomain_precision = []
    mcdonald_precision = []
    sonic_precision = []
    indomain_recall = []
    mcdonald_recall = []
    sonic_recall = []
    indomain_f1 = []
    mcdonald_f1 = []
    sonic_f1 = []

    device = torch.device('cuda:0')
    batch_size = args.bs

    model = model.to(device)

    dataPath = args.dataPath
    # if os.path.exists(f"../Data/{dataPath}_test.pkl"):
    #     testDataset = Dataset(pd.DataFrame(), 'test', dataPath)
    # else:
    #     dirPath = '../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
    #     data = pd.read_csv(dirPath)
    #     threshold = data['funny_score_y'].quantile(0.75)
    #     data = data[data['funny_score_y'] >= threshold]
    #     unique_image_ids = data['image_id'].unique()
    #     # train_ids, test_ids = train_test_split(unique_image_ids, test_size=0.2, random_state=42)
    #     # train = data[data['image_id'].isin(train_ids)]
    #     # test = data[data['image_id'].isin(test_ids)]
    #     # print(train.shape, test.shape)
    #     # testDataset = Dataset(test, 'test', dataPath)
    #     testDataset = Dataset(data, 'test', dataPath)

    if os.path.exists(f"../Data/{dataPath}_mcdonalds_switzerland.pkl"):
        mcdonaldDataset = InsDataset(pd.DataFrame(), 'mcdonalds_switzerland', dataPath)
    else:

        data = pd.read_csv('../Data/Instagram/Generate_mcdonalds_switzerland.csv')
        print("shape of data: ", data.shape)
        image_id_counts = data['image_id'].value_counts()
        threshold = 12
        original = pd.read_csv('../Data/Instagram/Filter_mcdonalds_switzerland.csv')
        original['caption'] = original['caption'].str.lower()
        data['caption'] = data['caption'].str.lower()
        original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
        original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)
        original = original[original['gen_count'] >= 100]
        #############################################
        #################   train   #################
        #############################################
        original_train = original[original['text_len'] >= threshold]
        # original_train = original[:621]
        original_train['funny_score'] = original_train['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        ################################################
        print(f'MCD: {original_train.shape}')
        mcdonaldDataset = InsDataset(original_train, 'mcdonalds_switzerland', dataPath)
        #############################################
        #################    test   #################
        #############################################
        # original_test = original[original['text_len'] < threshold]
        # # original_test = original[~original['image_id'].isin(original_train['image_id'])]
        # ins_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:(len(original_train['image_id'].unique()) // 4)]
        # ins_test['funny_score'] = ins_test['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        # ################################################
        # print(f'MCD: {ins_test.shape}')
        # mcdonaldDataset = InsDataset(ins_test, 'mcdonalds_switzerland', dataPath)
        ################################################
    if os.path.exists(f"../Data/{dataPath}_sonicdrivein.pkl"):
        sonicDataset = InsDataset(pd.DataFrame(), 'sonicdrivein', dataPath)
    else:
        data = pd.read_csv('../Data/Instagram/Generate_sonicdrivein.csv')
        print("shape of data: ", data.shape)
        image_id_counts = data['image_id'].value_counts()
        threshold = 10
        original = pd.read_csv('../Data/Instagram/Filter_sonicdrivein.csv')
        original['caption'] = original['caption'].str.lower()
        data['caption'] = data['caption'].str.lower()
        original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
        original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)
        original = original[original['gen_count'] >= 100]
        #############################################
        #################   train   #################
        #############################################
        original_train = original[original['text_len'] >= threshold]
        # original_train = original[:1136]
        original_train['funny_score'] = original_train['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        ################################################
        print(f'SD: {original_train.shape}')
        sonicDataset = InsDataset(original_train, 'sonicdrivein', dataPath)
        #############################################
        #################    test   #################
        #############################################
        # original_test = original[original['text_len'] < threshold]
        # # original_test = original[~original['image_id'].isin(original_train['image_id'])]
        # ins_test = original_test.sort_values(by=['funny_score'], ascending=[False])#[:186]#[ :(len(original_train['image_id'].unique()) // 4)]
        # ins_test['funny_score'] = ins_test['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        # ################################################
        # print(f'SD: {ins_test.shape}')
        # sonicDataset = InsDataset(ins_test, 'sonicdrivein', dataPath)
        ################################################
    # get data size
    # test_dataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)
    mcdonald_dataloader = DataLoader(mcdonaldDataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    sonic_dataloader = DataLoader(sonicDataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    # print(len(test_dataloader), len(mcdonald_dataloader), len(sonic_dataloader))
    print(len(mcdonald_dataloader), len(sonic_dataloader))

    f1_ = torchmetrics.F1Score(task='multiclass', num_classes=2, average='weighted')
    precision_ = torchmetrics.Precision(task='multiclass', num_classes=2, average='weighted')
    recall_ = torchmetrics.Recall(task='multiclass', num_classes=2, average='weighted')
    loss_class = FocalContrastiveLoss(output_dir, output_prefix, 'TestOnTest')
    for i in range(20):
        if os.path.exists(f'./Model/{output_dir}/checkpoint-{i + 1:03d}.pt'):
            model.load_state_dict(torch.load(f'./Model/{output_dir}/checkpoint-{i + 1:03d}.pt'))
            model = model.eval()
            model = model.to(device)
            # criterion = nn.BCELoss()
            # criterion = nn.MSELoss()
            # mae = nn.L1Loss()
            print(f">>> epoch {i + 1}")
            sys.stdout.flush()
            model.eval()
            with torch.no_grad():
                #     progress = tqdm(total=len(test_dataloader), desc=output_prefix)
                #     epoch_loss = 0
                #     epoch_accuracy = 0
                #     epoch_tp = 0
                #     epoch_accuracy_rank = 0
                #     mae_loss = 0
                #     output_df = pd.DataFrame(columns=['image_id', 'caption', 'groundtruth', 'humor_score'])
                #     for idx, (images, caption_ids, caption_masks, humor, rank, img_id, caption) in enumerate(test_dataloader):
                #         model.zero_grad()
                #         images, caption_ids, caption_masks, humor = images.to(device), caption_ids.to(device), caption_masks.to(device), humor.to(device)
                #         humor = humor.unsqueeze(1).to(device, dtype=torch.float)
                #         rank = rank.unsqueeze(1).to(device, dtype=torch.float)
                #         outputs = model(images, caption_ids, caption_masks)
                #         # loss = criterion(outputs, humor)
                #         # mae_loss += mae(outputs, humor).item()
                #         loss = criterion(outputs, rank)
                #         mae_loss += mae(outputs, rank).item()
                #         epoch_accuracy += (outputs >= 0.5).eq(rank >= 0.5).sum().item()
                #         epoch_tp += ((outputs >= 0.5).eq(rank >= 0.5)).eq(outputs >= 0.5).sum().item()
                #         epoch_accuracy_rank += ((outputs <=0.33) & (rank <= 0.33)).eq(torch.ones_like(rank)).sum().item()
                #         epoch_accuracy_rank += ((outputs > 0.33) & (outputs <= 0.66) & (rank > 0.33) & (rank <= 0.66)).eq(torch.ones_like(rank)).sum().item()
                #         epoch_accuracy_rank += ((outputs > 0.66) & (rank > 0.66)).eq(torch.ones_like(rank)).sum().item()
                #         output_df = pd.concat([output_df, pd.DataFrame({'image_id': img_id, 'caption': caption, 'groundtruth': humor.cpu().numpy().flatten(), 'rank': rank.cpu().numpy().flatten(), 'humor_score': outputs.cpu().numpy().flatten()})], axis=0)
                #         epoch_loss += loss.item()
                #         progress.update()
                #     epoch_loss /= len(test_dataloader)
                #     epoch_accuracy /= len(testDataset)
                #     epoch_tp /= len(testDataset)
                #     epoch_accuracy_rank /= len(testDataset)
                #     mae_loss /= len(test_dataloader)
                #     indomain_losses.append(epoch_loss)
                #     indomain_accuracy.append(epoch_accuracy)
                #     indomain_tp.append(epoch_tp)
                #     indomain_accuracy_rank.append(epoch_accuracy_rank)
                #     indomain_mae.append(mae_loss)
                #     progress.set_postfix({"loss": epoch_loss, "accuracy": epoch_accuracy, "mae": mae_loss})
                #     # indomain_accuracy.append(mae_loss)
                #     # progress.set_postfix({"loss": epoch_loss, "mae": mae_loss})
                #     progress.close()
                #     output_df['humor_score_result'] = output_df['humor_score'].apply(lambda x: 1 if x >= 0.5 else 0)
                #     output_df = output_df.reset_index(drop=True)
                #     output_df.to_csv(f"./Model/{output_dir}/oxford_test_{i + 1:03d}.csv", index=False)

                output_df = pd.DataFrame(columns=['image_id', 'caption', 'groundtruth', 'humor_score'])
                progress = tqdm(total=len(mcdonald_dataloader), desc=output_prefix)
                epoch_loss = 0
                # mae_loss = 0
                for idx, (images, caption_ids, caption_masks, humor, rank, img_id, caption) in enumerate(mcdonald_dataloader):
                    model.zero_grad()
                    images, caption_ids, caption_masks, rank = images.to(device), caption_ids.to(device), caption_masks.to(device), rank.to(device)
                    outputs, sim = model(images, caption_ids, caption_masks)
                    loss = loss_class.computeLoss(outputs, rank, sim)
                    # loss = criterion(outputs, humor)
                    # mae_loss += mae(outputs, humor).item()
                    # mae_loss += mae(outputs, rank).item()
                    output_df = pd.concat([output_df, pd.DataFrame(
                        {'image_id': img_id, 'caption': caption, 'groundtruth': humor.cpu().numpy().flatten(),
                         'rank': rank.cpu().numpy().flatten(), 'humor_score': outputs.cpu().numpy().flatten()})],
                                          axis=0)
                    epoch_loss += loss.item()
                    progress.update()
                output_df['humor_score_result'] = output_df['humor_score'].apply(lambda x: 1 if x >= 0.5 else 0)
                output_df = output_df.reset_index(drop=True)
                output_df.to_csv(f"./Model/{output_dir}/mcdonald_test_{i + 1:03d}.csv", index=False)

                epoch_loss /= len(mcdonald_dataloader)

                labels = torch.tensor(output_df['rank'].tolist())
                preds = torch.tensor(output_df['humor_score_result'].tolist())

                epoch_tp = ((preds >= 0.5) & (labels >= 0.5)).sum().item()
                epoch_tn = ((preds < 0.5) & (labels < 0.5)).sum().item()
                epoch_fn = ((preds < 0.5) & (labels >= 0.5)).sum().item()
                epoch_fp = ((preds >= 0.5) & (labels < 0.5)).sum().item()
                precision = precision_(preds, labels)
                recall = recall_(preds, labels)
                f1 = f1_(preds, labels)
                accuracy = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)

                mcdonald_losses.append(epoch_loss)
                mcdonald_tp.append(epoch_tp)
                mcdonald_tn.append(epoch_tn)
                mcdonald_fp.append(epoch_fp)
                mcdonald_fn.append(epoch_fn)
                # mcdonald_mae.append(mae_loss)
                mcdonald_precision.append(precision.item())
                mcdonald_recall.append(recall.item())
                mcdonald_f1.append(f1.item())
                mcdonald_accuracy.append(accuracy)
                # mcdonald_accuracy.append(mae_loss)


                output_df = pd.DataFrame(columns=['image_id', 'caption', 'groundtruth', 'humor_score'])
                progress = tqdm(total=len(sonic_dataloader), desc=output_prefix)
                epoch_loss = 0
                epoch_accuracy = 0
                epoch_tp = 0
                epoch_fp = 0
                epoch_fn = 0
                epoch_tn = 0
                epoch_accuracy_rank = 0
                # mae_loss = 0
                for idx, (images, caption_ids, caption_masks, humor, rank, img_id, caption) in enumerate(
                        sonic_dataloader):
                    model.zero_grad()
                    images, caption_ids, caption_masks, rank = images.to(device), caption_ids.to(device), caption_masks.to(device), rank.to(device)
                    humor = humor.unsqueeze(1).to(device, dtype=torch.float)
                    outputs, sim = model(images, caption_ids, caption_masks)
                    loss = loss_class.computeLoss(outputs, rank, sim)
                    # mae_loss += mae(outputs, rank).item()
                    output_df = pd.concat([output_df, pd.DataFrame(
                        {'image_id': img_id, 'caption': caption, 'groundtruth': humor.cpu().numpy().flatten(),
                         'rank': rank.cpu().numpy().flatten(), 'humor_score': outputs.cpu().numpy().flatten()})],
                                          axis=0)
                    epoch_loss += loss.item()
                    progress.update()
                output_df['humor_score_result'] = output_df['humor_score'].apply(lambda x: 1 if x >= 0.5 else 0)
                output_df = output_df.reset_index(drop=True)
                output_df.to_csv(f"./Model/{output_dir}/sonic_test_{i + 1:03d}.csv", index=False)

                epoch_loss /= len(sonic_dataloader)

                labels = torch.tensor(output_df['rank'].tolist())
                preds = torch.tensor(output_df['humor_score_result'].tolist())

                epoch_tp = ((preds >= 0.5) & (labels >= 0.5)).sum().item()
                epoch_tn = ((preds < 0.5) & (labels < 0.5)).sum().item()
                epoch_fn = ((preds < 0.5) & (labels >= 0.5)).sum().item()
                epoch_fp = ((preds >= 0.5) & (labels < 0.5)).sum().item()
                precision = precision_(preds, labels)
                recall = recall_(preds, labels)
                f1 = f1_(preds, labels)
                accuracy = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)

                sonic_losses.append(epoch_loss)

                sonic_tp.append(epoch_tp)
                sonic_tn.append(epoch_tn)
                sonic_fp.append(epoch_fp)
                sonic_fn.append(epoch_fn)
                # sonic_mae.append(mae_loss)
                sonic_precision.append(precision.item())
                sonic_recall.append(recall.item())
                sonic_f1.append(f1.item())
                sonic_accuracy.append(accuracy)
                # sonic_accuracy.append(mae_loss)


            loss_data = pd.DataFrame()
            # loss_data['indomain_loss'] = indomain_losses
            loss_data['mcdonald_loss'] = mcdonald_losses
            loss_data['sonic_loss'] = sonic_losses
            # loss_data['indomain_accuracy'] = indomain_accuracy
            loss_data['mcdonald_accuracy'] = mcdonald_accuracy
            loss_data['sonic_accuracy'] = sonic_accuracy
            # loss_data['indomain_precision'] = indomain_precision
            loss_data['mcdonald_precision'] = mcdonald_precision
            loss_data['sonic_precision'] = sonic_precision
            # loss_data['indomain_recall'] = indomain_recall
            loss_data['mcdonald_recall'] = mcdonald_recall
            loss_data['sonic_recall'] = sonic_recall
            # loss_data['indomain_f1'] = indomain_f1
            loss_data['mcdonald_f1'] = mcdonald_f1
            loss_data['sonic_f1'] = sonic_f1
            # loss_data['indomain_tp'] = indomain_tp
            loss_data['mcdonald_tp'] = mcdonald_tp
            loss_data['sonic_tp'] = sonic_tp
            # loss_data['indomain_tn'] = indomain_tn
            loss_data['mcdonald_tn'] = mcdonald_tn
            loss_data['sonic_tn'] = sonic_tn
            # loss_data['indomain_fp'] = indomain_fp
            loss_data['mcdonald_fp'] = mcdonald_fp
            loss_data['sonic_fp'] = sonic_fp
            # loss_data['indomain_fn'] = indomain_fn
            loss_data['mcdonald_fn'] = mcdonald_fn
            loss_data['sonic_fn'] = sonic_fn

            loss_data.to_csv(f"./Model/{output_dir}/{output_prefix}_test.csv", index=False)

def test_mine(model, args, dataform: str = 'Oxford', input_num: int = 5, model_path: str = ''):

    device = torch.device('cuda:0')

    model = model.to(device)
    model.eval()
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn’t have a pad token by default
    # Define image transform
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    loss_class = FocalContrastiveLoss(args.out_dir, args.prefix, 'TestOnTest')
    df = pd.DataFrame()
    with torch.no_grad():
        for n in range(input_num):
            # dataPath = f'./Model/{args.out_dir}/{args.out_dir}_test_{n + 1:03d}.csv'
            if os.path.exists(f'./Model/{args.out_dir}/{args.out_dir}_test_{n + 1:03d}.csv'):
                dataPath = f'./Model/{args.out_dir}/{args.out_dir}_test_{n + 1:03d}.csv'
                save_path = f'./Model/{args.out_dir}/'
            elif os.path.exists(f'./Model/{args.out_dir}/test_{n + 1:03d}.csv'):
                dataPath = f'./Model/{args.out_dir}/test_{n + 1:03d}.csv'
                save_path = f'./Model/{args.out_dir}/'
            elif dataform == 'Oxford':
                save_path = f'./Model/{args.out_dir}/oxford/'
                if os.path.exists(f'{save_path}{args.out_dir}_test_{n + 1:03d}.csv'):
                    dataPath = f'{save_path}{args.out_dir}_test_{n + 1:03d}.csv'
                elif os.path.exists(f'{save_path}test_{n + 1:03d}.csv'):
                    dataPath = f'{save_path}test_{n + 1:03d}.csv'
                else:
                    dataPath = "none"
            if dataform != 'Oxford':
                save_path = f'./Model/{args.out_dir}/test/'
                # save_path = f'../Citations/CLIP_prefix_caption/Model/{args.out_dir}/'#ins/'
                if os.path.exists(f'{save_path}{args.which}_{n + 1}.csv'):
                    dataPath = f'{save_path}{args.which}_{n + 1}.csv'

                # if os.path.exists(f'{save_path}{args.out_dir}_test_{n + 1:03d}.csv'):
                #     dataPath = f'{save_path}{args.out_dir}_test_{n + 1:03d}.csv'
                # elif os.path.exists(f'{save_path}test_{n + 1:03d}.csv'):
                #     dataPath = f'{save_path}test_{n + 1:03d}.csv'
                else:
                    dataPath = "none"
            else:
                dataPath = "none"

            if dataPath != "none":
                print(dataPath)
                data = pd.read_csv(dataPath)
                image_list = data['image_id'].tolist()
                # image_list = []
                # image_list.append('')
                # image_list += args.test_image_id_list[:10] + args.train_image_id_list[:10]
                # image_list.append('-')
                # image_list.append('-')
                # image_list = image_list * 3
                # data['image_id'] = image_list

                # caption_list = data['text'].tolist()
                caption_list = data['caption'].tolist()
                data['humor_score'] = ''
                print(len(image_list), len(caption_list))
                if dataform == 'Oxford':
                    image_path = '../Data/Oxford_HIC/oxford_img/'
                else:
                    image_path = f'../Data/Instagram/sonicdrivein_img/'
                    image_path2 = f'../Data/Instagram/mcdonalds_switzerland_img/'

                for i in range(len(image_list)):
                    if str(image_list[i]) != 'nan' and str(image_list[i]) != '-' and str(image_list[i]) != '':
                        if os.path.exists(f'{image_path}{image_list[i]}.jpg'):
                            filename = f"{image_path}{image_list[i]}.jpg"
                        else:
                            filename = f"{image_path2}{image_list[i]}.jpg"
                        image = Image.open(filename).convert('RGB')
                        image = image_transform(image)
                        inputs = tokenizer([caption_list[i]], truncation=True, max_length=64, return_tensors="pt")
                        input_ids = inputs["input_ids"]
                        attention_mask = inputs["attention_mask"]
                        padding = torch.zeros(1, 64 - input_ids.shape[1], dtype=torch.int64)
                        input_ids = torch.cat((input_ids, padding), dim=1).squeeze(0)
                        attention_mask = torch.cat((attention_mask, padding), dim=1).squeeze(0)

                        images, caption_ids, caption_masks = image.unsqueeze(0).to(device), input_ids.unsqueeze(0).to(device), attention_mask.unsqueeze(0).to(device)
                        outputs, humor_sim, non_humor_sim = model(images, caption_ids, caption_masks)
                        data.loc[i, 'humor_score'] = outputs.item()
                    else:
                        if str(image_list[i]) == 'nan':
                            data.loc[i, 'humor_score'] = ''
                        else:
                            data.loc[i, 'humor_score'] = image_list[i]
                data['humor'] = data['humor_score'].apply(lambda x: x if type(x) != float else (1 if x > 0.5 else 0))
                if df.empty:
                    df = data[['image_id', 'caption', 'humor_score','humor']]
                else:
                    df = pd.concat([df, data[['image_id', 'caption', 'humor_score','humor']]], axis=1)
        # df.to_csv(f'{save_path}{args.prefix}_{model_path}_test_humorscore.csv', float_format='%.15f', index=False)
        df.to_csv(f'{save_path}0409_SD_MC_focalLoss_09{model_path}_{args.which}_humorscore.csv', float_format='%.15f', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default='humorScore/humorScore_SD_pass10_MC_pass12_noAug_wOxford_only01_train', help='data path')
    # parser.add_argument('--dataPath', default='humorScore_oxford_with_coco_only01',help='data path')
    parser.add_argument('--out_dir', default='humorScore/humorScore_20250418_SD_pass10_MC_pass12_noAug_wOxford_only01_focal09_16_contrast_temp040')
    ### test_mine
    # parser.add_argument('--out_dir', default='20250217_sonicdrivein_only300_base_oxford_lower_only800_transformer_onlyLLMlora_p64_falcon_swin_tf8')
    # parser.add_argument('--out_dir', default='20250408_100up_only200_lessNotFunImg_169_279_passlength_10_SD_base_0316_oxford_800_8_300_2_ESH_filter_cross_concat_combineLoss')
    # parser.add_argument('--out_dir', default='20250409_100up_only200_lessNotFunImg_53_171_passlength_12_MC_base_0316_oxford_800_8_300_2_ESH_filter_cross_concat_combineLoss')
    # parser.add_argument('--out_dir',default='clip_100up_only200_lessNotFunImg_53_171_passlength_12_MC_base_oxford_800_8_300_82')
    # parser.add_argument('--out_dir',default='clip_100up_only200_lessNotFunImg_169_279_passlength_10_base_oxford_800_8_300_82')
    # parser.add_argument('--out_dir', default='humorScore_20250409_SD_only100_pass10_MC_only200_pass12_funnyscore_only01_focalLoss_09')
    parser.add_argument('--prefix', default='checkpoint', help='prefix for saved filenames')
    # parser.add_argument('--prefix', default='humorScore_20250409_SD_only100_pass10_MC_only200_pass12_funnyscore_only01_focalLoss_09', help='prefix for saved filenames')
    parser.add_argument('--which', default='generate_beam')
    # parser.add_argument('--which', default='generate2')

    parser.add_argument('--bs', type=int, default=90)
    # parser.add_argument('--bs', type=int, default=1500)
    args = parser.parse_args()
    if not os.path.exists('./Model/' + args.out_dir):
        os.makedirs('./Model/' + args.out_dir)
        os.makedirs('D:/MemeGAN/Model/' + args.out_dir)

    model = ImageTextModel()
    device = torch.device('cuda:0')
    model = model.to(device)
    # save_file = 'humorScore_20250314_oxford_with_coco_funnyscore_rank_MSE'
    # i = 3
    model.eval()
    # train(model, args, output_dir=args.out_dir, output_prefix=args.prefix)
    test(model, args, output_dir=args.out_dir, output_prefix=args.prefix)
    # for i in range(20):
    #     i = i + 5
    #     if os.path.exists(f'./Model/{args.prefix}/checkpoint-{i:03d}.pt'):
    #         model.load_state_dict(torch.load(f'./Model/{args.prefix}/checkpoint-{i:03d}.pt'))
    #         # test_mine(model, args, dataform="Oxford", input_num=10, model_path = i)
    #         test_mine(model, args, dataform="sonicdrivein", input_num=10, model_path=i)

    old_test = -1
    trainData = '../Data/Oxford_HIC/parse/oxford_lower_800up_only800_all_ViT-B_32_train.pkl'
    testData = '../Data/Oxford_HIC/parse/oxford_lower_800up_only800_rest_300up_top300_ViT-B_32_test.pkl'
    # trainData = '../Data/Instagram/parse/300up_only300_all_sonicdrivein_ViT-B_32_train.pkl'
    # testData = '../Data/Instagram/parse/100up_only100_rest_50up_top50_sonicdrivein_ViT-B_32_test.pkl'
    # trainData = '../Data/Instagram/parse/300up_only300_all_sonicdrivein_ViT-B_32_train.pkl'
    # testData = '../Data/Instagram/parse/300up_only300_rest_200up_top200_sonicdrivein_ViT-B_32_test.pkl'
    train_dataform = "sonicdrivein"
    test_dataform = "sonicdrivein"
    if old_test > 0:
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

            def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Any, int]:
                tokens, mask = self.pad_tokens(item)
                if self.dataFrom == 'Oxford':
                    prefix = torch.load('../../Oxford_HIC/ImageData/' + self.image_ids[item] + '.pt',
                                        weights_only=False)
                else:
                    prefix = torch.load(
                        '../../Instagram/ImageData/' + self.dataFrom + '/' + self.image_ids[item] + '.pt',
                        weights_only=False)
                # emotion, sentiment, humor = self.pad_emotion(item)
                if self.normalize_prefix:
                    prefix = prefix.float()
                    prefix = prefix / prefix.norm(2, -1)
                return tokens, mask, prefix, item  # , emotion, sentiment, humor

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
                    prefix_embeds = model.clip_project(prefixes_batch).view(-1, self.prefix_length,
                                                                            model.embedding_size)
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
                print(
                    f"Before Filtered: {len(self.captions_tokens)}, After Filtered: {len(filtered_indices)}, BLEU <= {bleu_threshold}")
                self.captions = [self.captions[i] for i in filtered_indices]
                self.image_ids = [self.image_ids[i] for i in filtered_indices]
                self.captions_tokens = [self.captions_tokens[i] for i in filtered_indices]
                self.caption2embedding = [self.caption2embedding[i] for i in filtered_indices]
                del tokens_batch, masks_batch, prefixes_batch, original_indices, prefix_embeds, text_embeds, embedding_cat, outputs, logits, generated_tokens_batch, reference, candidate, bleu_score
                gc.collect()
                torch.cuda.empty_cache()

            def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2", normalize_prefix=False,
                         model=None,
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
                    self.emotion[image_id] = torch.tensor(
                        self.tokenizer.encode(emotion, max_length=64, truncation=True),
                        dtype=torch.int64)
                    self.sentiment[image_id] = torch.tensor(
                        self.tokenizer.encode(sentiment, max_length=64, truncation=True), dtype=torch.int64)
                    self.humor[image_id] = torch.tensor(
                        self.tokenizer.encode(humor, max_length=64, truncation=True, padding=True), dtype=torch.int64)

                print(f"Train Data size: {len(self.captions_tokens)}")
                # self.filter_data_by_bleu(model, batch_size, bleu_threshold)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prefix_length = 64
        normalize_prefix = False
        trainDataset = OxfordDataset(trainData, prefix_length, normalize_prefix=normalize_prefix, dataFrom=train_dataform)
        testDataset = OxfordDataset(testData, prefix_length, normalize_prefix=normalize_prefix, dataFrom=test_dataform)

        if train_dataform:
            ####################### default #######################
            train_image = []
            train_text = []
            for i in range(len(trainDataset)):
                caption = trainDataset.captions[i]
                image_id = trainDataset.image_ids[i]
                if image_id not in train_image:
                    print(f"Image ID: {image_id}, Caption: {caption}")
                    train_image.append(image_id)
                    train_text.append(caption)
                    print(
                        f"Emotion: {trainDataset.emotion[image_id].shape}, Sentiment: {trainDataset.sentiment[image_id].shape}, Humor: {trainDataset.humor[image_id].shape}")
                    if len(train_image) == 10:
                        break
            #######################################################
            tokens_list = []
            mask_list = []
            prefix_list = []
            # train_emotion_list = []
            # train_sentiment_list = []
            # train_humor_list = []
            train_gt = []
            train_caption = dict()
            train_image_id_list = []
            if train_dataform != "Oxford":
                load = pd.read_csv(f'../Data/Instagram/CaptionID_{train_dataform}.csv')
                load['caption'] = load['caption'].str.lower()
                train_text = []
                for i in range(len(train_image)):
                    caption = load[load['image_id'] == train_image[i]]['caption'].values[0]
                    train_text.append(caption)
                inside = []
                for i in range(len(trainDataset)):
                    caption = trainDataset.captions[i]
                    image_id = trainDataset.image_ids[i]
                    if image_id in train_image and image_id not in train_image_id_list:
                        train_caption[image_id] = []
                        train_caption[image_id].append(caption)
                        tokens, mask, prefix, item = trainDataset[i]
                        tokens_list.append(tokens)
                        mask_list.append(mask)
                        prefix_list.append(prefix)
                        emotion, sentiment, humor = trainDataset.pad_emotion(item)
                        # train_emotion_list.append(emotion)
                        # train_sentiment_list.append(sentiment)
                        # train_humor_list.append(humor)
                        train_gt.append(caption)

                        train_image_id_list.append(image_id)
            else:
                for i in range(len(trainDataset)):
                    caption = trainDataset.captions[i]
                    image_id = trainDataset.image_ids[i]
                    if image_id in train_image:
                        if train_caption.get(image_id) is not None:
                            train_caption[image_id].append(caption)
                        else:
                            train_caption[image_id] = []
                            train_caption[image_id].append(caption)
                        if caption in train_text and image_id not in train_image_id_list:
                            # tokens, mask, prefix, item, emotion, sentiment, humor = trainDataset[i]
                            tokens, mask, prefix, item = trainDataset[i]
                            tokens_list.append(tokens)
                            mask_list.append(mask)
                            prefix_list.append(prefix)
                            train_gt.append(caption)
                            # emotion, sentiment, humor = trainDataset.pad_emotion(item)
                            # train_emotion_list.append(emotion)
                            # train_sentiment_list.append(sentiment)
                            # train_humor_list.append(humor)
                            train_image_id_list.append(image_id)
            train_tokens = torch.stack(tokens_list).to(device)
            train_mask = torch.stack(mask_list).to(device)
            train_prefix = torch.stack(prefix_list).to(device)
            # train_emotion = torch.stack(train_emotion_list).to(device)
            # train_sentiment = torch.stack(train_sentiment_list).to(device)
            # train_humor = torch.stack(train_humor_list).to(device)
            print(train_tokens.shape, train_mask.shape, train_prefix.shape, len(train_image_id_list))
            # print(train_emotion.shape, train_sentiment.shape, train_humor.shape)
        if test_dataform:
            ####################### default #######################
            test_image = []
            test_text = []
            # test_emotion_list = []
            # test_sentiment_list = []
            # test_humor_list = []
            for i in range(len(testDataset)):
                caption = testDataset.captions[i]
                image_id = testDataset.image_ids[i]
                if image_id not in test_image:
                    print(f"Image ID: {image_id}, Caption: {caption}")
                    test_image.append(image_id)
                    test_text.append(caption)
                    if len(test_image) == 10:
                        break
            #######################################################
            tokens_list = []
            mask_list = []
            prefix_list = []
            # test_emotion_list = []
            # test_sentiment_list = []
            # test_humor_list = []
            test_gt = []
            test_caption = dict()
            test_image_id_list = []

            if test_dataform != "Oxford":
                load = pd.read_csv(f'../Data/Instagram/CaptionID_{test_dataform}.csv')
                load['caption'] = load['caption'].str.lower()
                test_text = []
                for i in range(len(test_image)):
                    caption = load[load['image_id'] == test_image[i]]['caption'].values[0]
                    test_text.append(caption)
                for i in range(len(testDataset)):
                    caption = testDataset.captions[i]
                    image_id = testDataset.image_ids[i]
                    if image_id in test_image and image_id not in test_image_id_list:
                        test_caption[image_id] = []
                        test_caption[image_id].append(caption)
                        tokens, mask, prefix, item = testDataset[i]
                        tokens_list.append(tokens)
                        mask_list.append(mask)
                        prefix_list.append(prefix)
                        # emotion, sentiment, humor = testDataset.pad_emotion(item)
                        # test_emotion_list.append(emotion)
                        # test_sentiment_list.append(sentiment)
                        # test_humor_list.append(humor)
                        test_gt.append(caption)
                        test_image_id_list.append(image_id)
            else:
                for i in range(len(testDataset)):
                    caption = testDataset.captions[i]
                    image_id = testDataset.image_ids[i]
                    if image_id in test_image:
                        if test_caption.get(image_id) is not None:
                            test_caption[image_id].append(caption)
                        else:
                            test_caption[image_id] = []
                            test_caption[image_id].append(caption)
                        if caption in test_text and image_id not in test_image_id_list:
                            # print(f"Image ID: {image_id}, Caption: {caption}")
                            tokens, mask, prefix, item = testDataset[i]
                            tokens_list.append(tokens)
                            mask_list.append(mask)
                            prefix_list.append(prefix)
                            # emotion, sentiment, humor = testDataset.pad_emotion(item)
                            # test_emotion_list.append(emotion)
                            # test_sentiment_list.append(sentiment)
                            # test_humor_list.append(humor)
                            test_gt.append(caption)
                            test_image_id_list.append(image_id)

            test_tokens = torch.stack(tokens_list).to(device)
            test_mask = torch.stack(mask_list).to(device)
            test_prefix = torch.stack(prefix_list).to(device)
            # test_emotion = torch.stack(test_emotion_list).to(device)
            # test_sentiment = torch.stack(test_sentiment_list).to(device)
            # test_humor = torch.stack(test_humor_list).to(device)
            print(test_tokens.shape, test_mask.shape, test_prefix.shape)
            # print(test_emotion.shape, test_sentiment.shape, test_humor.shape)

        args.test_image_id_list = test_image_id_list
        args.train_image_id_list = train_image_id_list
        test_mine(model, args, dataform=train_dataform, input_num=old_test)

if __name__ == '__main__':
    main()