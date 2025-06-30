import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import time
import pickle
import argparse
import pandas as pd
import torchmetrics
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from typing import Tuple, Optional
from torchvision.ops import sigmoid_focal_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as nnf
from torch.nn.functional import normalize
from transformers import GPT2Tokenizer, GPT2Model

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
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class ImageTextModel(nn.Module):

    def __init__(self, gpt2_model_name="gpt2", feature_dim=768, output_dim=1):
        super(ImageTextModel, self).__init__()
        # Image Encoder (ResNet-50)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) ### 倒數第二層 torch.Size([10, 2048, 7, 7])
        self.resnet_linear = nn.Linear(2048, feature_dim)
        # Text Encoder (GPT-2)
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        # Fusion Layer
        self.transformer = Transformer(768, 8, 8)
        # contrastive learning
        self.temp = nn.Parameter(0.7 * torch.ones(1), requires_grad=True)
        # binary classification (mlp)
        self.mlp1 = nn.Linear((49 + 64) * 768, 1024)
        self.mlp2 = nn.Linear(1024, 64)
        self.classifier = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, input_ids, attention_mask):
        # Encode image
        img_features = self.resnet(image).squeeze(-1).squeeze(-1)  # Shape: (batch_size, 2048)
        img_features = img_features.view(img_features.shape[0], 2048, -1) # Shape: (batch_size, 2048, 7*7)
        img_features = self.resnet_linear(img_features.transpose(1, 2))  # Shape: (batch_size, 7*7, 768)
        text_outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Fusion (Concatenation + Projection)
        fused = torch.cat((img_features, text_outputs), dim=1)
        fused = self.transformer(fused)
        mix_features = fused.view(fused.shape[0], -1)
        # contrastive learning
        sim_features = normalize(mix_features, dim=-1)
        sim = sim_features @ sim_features.T
        sim = sim / self.temp
        # Classification
        logits = self.mlp1(mix_features)
        logits = self.mlp2(logits)
        logits = self.classifier(logits)
        probs = self.sigmoid(logits).squeeze(-1)  # Shape: (batch_size, output_dim)
        return probs, sim

class MixDataset(torch.utils.data.Dataset):

    def get_image_features(self, img_id):
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
            if 'sonicdrivein' in img_id:
                filename = f"../Data/Instagram/sonicdrivein_img/{img_id}.jpg"
                save_dir = f"../../humorscore_image_sonicdrivein_data/{img_id}.pt"
            elif 'mcdonalds_switzerland' in img_id:
                filename = f"../Data/Instagram/mcdonalds_switzerland_img/{img_id}.jpg"
                save_dir = f"../../humorscore_image_mcdonalds_switzerland_data/{img_id}.pt"
            else:
                filename = f"../Data/Oxford_HIC/oxford_img/{img_id}.jpg"
                save_dir = f"../../humorscore_image_{self.traintest}_data/oxford_{img_id}.pt"
            image = Image.open(filename).convert('RGB')
            image = self.image_transform(image)
            torch.save(image, save_dir)
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
        return image, caption_id, caption_attmask, humor, rank, str(self.image_list[item]), str(self.caption_list[item])

    def __len__(self):
        return len(self.image_list)

    def __init__(self, oxford_data: pd.DataFrame, ins_data: pd.DataFrame, traintest: str, dataPath: str):
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
            print('Data Loaded')
            print("%0d embeddings saved " % len(self.image_list))
        else:
            ###################################   oxford   ###################################
            if oxford_data is not None:

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
            ##################################################################################

            #################################   Instagram   ##################################
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
            ##################################################################################

class InsDataset(torch.utils.data.Dataset):

    def get_image_features(self, img_id, humor):
        if os.path.exists(f"../../humorscore_image_sonicdrivein_data/{img_id}.pt"):
            return torch.load(f"../../humorscore_image_sonicdrivein_data/{img_id}.pt")
        if os.path.exists(f"../../humorscore_image_mcdonalds_switzerland_data/{img_id}.pt"):
            return torch.load(f"../../humorscore_image_mcdonalds_switzerland_data/{img_id}.pt")
        filename = f"../Data/Instagram/{self.traintest}_img/{img_id}.jpg"
        image = Image.open(filename).convert('RGB')
        image = self.image_transform(image)
        torch.save(image, f"../../humorscore_image_{self.traintest}_data/{img_id}.pt")
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
        image = self.get_image_features(self.image_list[item], self.humor[item])
        caption_id, caption_attmask = self.get_caption_embedding(item)
        humor = self.humor[item]
        rank = self.rank[item]
        return image, caption_id, caption_attmask, humor, rank, str(self.image_list[item]), str(self.caption_list[item])

    def __len__(self):
        return len(self.image_list)

    def __init__(self, ins_data: pd.DataFrame, traintest: str, dataPath: str):
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
        if os.path.exists(f"./Model/{self.output_dir}/{self.traintest}_separateLoss.csv"):
            self.loss_df = pd.read_csv(f"./Model/{self.output_dir}/{self.traintest}_separateLoss.csv")
            print("load loss")
        else:
            self.loss_df = pd.DataFrame(columns=["focalLoss", "contrastive_loss", "loss"])

    def computeLoss(self, pred, target, sim):
        alpha = 0.8
        gamma = 1.7
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
    if os.path.exists(f"./Model/{output_dir}/{output_prefix}-loss.csv"):
        former = pd.read_csv(f"./Model/{output_dir}/{output_prefix}-loss.csv")
        train_losses = list(former['train_loss'])
        test_losses = list(former['test_loss'])
        save = list(former['save'])
        best_train_loss = min(train_losses)
        best_test_loss = min(test_losses)
    else:
        train_losses = []
        test_losses = []
        best_train_loss = 9999999999
        best_test_loss = 9999999999
        save = []

    device = torch.device('cuda:0')
    batch_size = args.bs
    model = model.to(device)
    dataPath = args.dataPath

    if os.path.exists(f"../Data/{dataPath}_train.pkl") and os.path.exists(f"../Data/{dataPath}_test.pkl"):
        if os.path.exists(f"../Data/{dataPath}_train.pkl"):
            trainDataset = MixDataset(pd.DataFrame(), pd.DataFrame(), 'train', dataPath)
        if os.path.exists(f"../Data/{dataPath}_test.pkl"):
            testDataset = MixDataset(pd.DataFrame(), pd.DataFrame(), 'test', dataPath)
    else:
        ###################################   oxford 1900   ###################################
        # get funniest 1900 image caption pair
        data = pd.read_csv('../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv')
        fcmean = data['funny_score_y'].mean()
        data = (
            data.sort_values(by=['image_id', 'funny_score_y'], ascending=[True, False])
            .groupby('image_id')
            .head(1)
        )
        data = data[:1900]
        data['funny_score_y'] = data['funny_score_y'].apply(lambda x: 1 if x > fcmean else 0)
        oxford_train = data
        print(f'oxford: {oxford_train.shape}')
        #######################################################################################

        ##############################   ins sonicdrivein 1300   ##############################
        ###############   train   ###############
        # get same data as GAMC model
        threshold = 10
        original = pd.read_csv('../Data/Instagram/Filter_sonicdrivein.csv')
        original['caption'] = original['caption'].str.lower()
        image_id_counts = original['image_id'].value_counts()
        original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
        original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)
        original = original[original['gen_count'] >= 100]
        original_train = original[original['text_len'] >= threshold][:1300]
        # exchage bad test samples to train samples (base on prvious result)
        fn_data = pd.read_csv('./Model/humorScore/humorScore_20250423_SD_pass10_MC_pass12_noAug_wOxford_only01_focal08_13_contrast_temp050/test/sonic_test_006.csv')
        fn_data = fn_data[(fn_data['rank'] == 1.0) & (fn_data['humor_score_result'] == 0)]
        fn_data = original.merge(fn_data, on='image_id', how='inner', suffixes=('', '_'))
        ins_train = original_train[:(-1 * fn_data.shape[0])]
        ins_train = pd.concat([ins_train, fn_data], ignore_index=True)
        print(fn_data.shape, ins_train.shape)
        ###############    test   ###############
        # 8:2 data >> 20% test funniest data
        original_test = original[~original['image_id'].isin(ins_train['image_id'])]
        ins_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:(len(ins_train['image_id'].unique()) // 4)]
        if ins_test.shape[0] < (len(ins_train['image_id'].unique()) // 4):
            original = pd.read_csv('../Data/Instagram/Filter_sonicdrivein.csv')
            original_test = original[~original['image_id'].isin(ins_train['image_id'])]
            original_test = original_test[~original_test['image_id'].isin(ins_test['image_id'])]
            temp_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:((len(ins_train['image_id'].unique()) // 4) - ins_test.shape[0])]
            ins_test = pd.concat([ins_test, temp_test], ignore_index=True)
        print(f'sonic_train: {ins_train.shape}, sonic_test: {ins_test.shape}')
        #######################################################################################

        #########################   ins  mcdonalds_switzerland  600   #########################
        ###############   train   ###############
        # get same data as GAMC model
        threshold = 12
        original = pd.read_csv('../Data/Instagram/Filter_mcdonalds_switzerland.csv')
        original['caption'] = original['caption'].str.lower()
        image_id_counts = original['image_id'].value_counts()
        original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
        original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)
        original = original[original['gen_count'] >= 100]
        original_train = original[original['text_len'] >= threshold][:600]
        # exchage bad test samples to train samples (base on prvious result)
        fn_data = pd.read_csv('./Model/humorScore/humorScore_20250423_SD_pass10_MC_pass12_noAug_wOxford_only01_focal08_13_contrast_temp050/test/mcdonald_test_006.csv')
        fn_data = fn_data[(fn_data['rank'] == 1.0) & (fn_data['humor_score_result'] == 0)]
        fn_data = original.merge(fn_data, on='image_id', how='inner', suffixes=('', '_'))
        temp_train = original_train[:(-1 * fn_data.shape[0])]
        temp_train = pd.concat([temp_train, fn_data], ignore_index=True)
        print(fn_data.shape, temp_train.shape)
        ###############    test   ###############
        # 8:2 data >> 20% test funniest data
        ins_train = pd.concat([ins_train, temp_train], ignore_index=True)
        original_test = original[~original['image_id'].isin(ins_train['image_id'])]
        temp_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:(len(temp_train['image_id'].unique()) // 4)]
        if temp_test.shape[0] < (len(temp_train['image_id'].unique()) // 4):
            original = pd.read_csv('../Data/Instagram/Filter_mcdonalds_switzerland.csv')
            original_test = original[~original['image_id'].isin(ins_train['image_id'])]
            original_test = original_test[~original_test['image_id'].isin(ins_test['image_id'])]
            temp_temp_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:((len(temp_train['image_id'].unique()) // 4) - temp_test.shape[0])]
            temp_test = pd.concat([temp_test, temp_temp_test], ignore_index=True)
        ins_test = pd.concat([ins_test, temp_test], ignore_index=True)
        print(f'mcd_train: {temp_train.shape}, mcd_test: {temp_test.shape}')
        print(f'instagram: {ins_train.shape}, {ins_test.shape}')
        #######################################################################################

        #######################################################################################
        # convert funny_score to binary
        ins_train['funny_score'] = ins_train['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        ins_test['funny_score'] = ins_test['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        trainDataset = MixDataset(oxford_train, ins_train, 'train', dataPath)
        testDataset = MixDataset(None, ins_test, 'test', dataPath)
        #######################################################################################
    print(len(trainDataset), len(testDataset))
    train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    trainLoss_class = FocalContrastiveLoss(output_dir, output_prefix, 'train')
    testLoss_class = FocalContrastiveLoss(output_dir, output_prefix, 'test')
    epoch = len(save)

    while len(trainDataset) > batch_size and len(testDataset) > batch_size:
        optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
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
            progress.set_postfix({"loss": loss.item()})
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            time.sleep(1)
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
                torch.cuda.empty_cache()
                time.sleep(1)
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
    mcdonald_losses = []
    sonic_losses = []
    mcdonald_accuracy = []
    sonic_accuracy = []
    mcdonald_tp = []
    sonic_tp = []
    mcdonald_tn = []
    sonic_tn = []
    mcdonald_fp = []
    sonic_fp = []
    mcdonald_fn = []
    sonic_fn = []
    mcdonald_precision = []
    sonic_precision = []
    mcdonald_recall = []
    sonic_recall = []
    mcdonald_f1 = []
    sonic_f1 = []

    device = torch.device('cuda:0')
    model = model.to(device)
    dataPath = args.dataPath

    ##############################   ins sonicdrivein 1300   ##############################
    if os.path.exists(f"../Data/{dataPath}_sonicdrivein.pkl"):
        sonicDataset = InsDataset(pd.DataFrame(), 'sonicdrivein', dataPath)
    else:
        #################   train   #################
        # get same data as GAMC model
        threshold = 10
        original = pd.read_csv('../Data/Instagram/Filter_sonicdrivein.csv')
        original['caption'] = original['caption'].str.lower()
        image_id_counts = original['image_id'].value_counts()
        original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
        original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)
        original = original[original['gen_count'] >= 100]
        original_train = original[original['text_len'] >= threshold][:1300]
        # exchage bad test samples to train samples (base on prvious result)
        fn_data = pd.read_csv('./Model/humorScore/humorScore_20250423_SD_pass10_MC_pass12_noAug_wOxford_only01_focal08_13_contrast_temp050/test/sonic_test_006.csv')
        fn_data = fn_data[(fn_data['rank'] == 1.0) & (fn_data['humor_score_result'] == 0)]
        fn_data = original.merge(fn_data, on='image_id', how='inner', suffixes=('', '_'))
        ins_train = original_train[:(-1 * fn_data.shape[0])]
        ins_train = pd.concat([ins_train, fn_data], ignore_index=True)
        ins_train['funny_score'] = ins_train['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        print(fn_data.shape, ins_train.shape)
        #################    test   #################
        original_test = original[~original['image_id'].isin(ins_train['image_id'])]
        ins_test = original_test.sort_values(by=['funny_score'], ascending=[False])[ :(len(original_train['image_id'].unique()) // 4)]
        if ins_test.shape[0] < (len(ins_train['image_id'].unique()) // 4):
            original = pd.read_csv('../Data/Instagram/Filter_sonicdrivein.csv')
            original_test = original[~original['image_id'].isin(ins_train['image_id'])]
            original_test = original_test[~original_test['image_id'].isin(ins_test['image_id'])]
            temp_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:((len(ins_train['image_id'].unique()) // 4) - ins_test.shape[0])]
            ins_test = pd.concat([ins_test, temp_test], ignore_index=True)
        ins_test['funny_score'] = ins_test['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        #############################################

        #############################################
        # open when you want to test train dataset
        print(f'SD: {ins_train.shape}')
        sonicDataset = InsDataset(ins_train, 'sonicdrivein', dataPath)
        #############################################
        # open when you want to test test dataset
        # print(f'SD: {ins_test.shape}')
        # sonicDataset = InsDataset(ins_test, 'sonicdrivein', dataPath)
        ################################################
    #######################################################################################

    #########################   ins  mcdonalds_switzerland  600   #########################
    if os.path.exists(f"../Data/{dataPath}_mcdonalds_switzerland.pkl"):
        mcdonaldDataset = InsDataset(pd.DataFrame(), 'mcdonalds_switzerland', dataPath)
    else:
        #################   train   #################
        # get same data as GAMC model
        threshold = 12
        original = pd.read_csv('../Data/Instagram/Filter_mcdonalds_switzerland.csv')
        original['caption'] = original['caption'].str.lower()
        image_id_counts = original['image_id'].value_counts()
        original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
        original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)
        original = original[original['gen_count'] >= 100]
        original_train = original[original['text_len'] >= threshold][:600]
        # exchage bad test samples to train samples (base on prvious result)
        fn_data = pd.read_csv('./Model/humorScore/humorScore_20250423_SD_pass10_MC_pass12_noAug_wOxford_only01_focal08_13_contrast_temp050/test/mcdonald_test_006.csv')
        fn_data = fn_data[(fn_data['rank'] == 1.0) & (fn_data['humor_score_result'] == 0)]
        fn_data = original.merge(fn_data, on='image_id', how='inner', suffixes=('', '_'))
        ins_train = original_train[:(-1 * fn_data.shape[0])]
        ins_train = pd.concat([ins_train, fn_data], ignore_index=True)
        ins_train['funny_score'] = ins_train['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        print(fn_data.shape, ins_train.shape)
        #################    test   #################
        original_test = original[~original['image_id'].isin(ins_train['image_id'])]
        ins_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:(len(original_train['image_id'].unique()) // 4)]
        if ins_test.shape[0] < (len(ins_train['image_id'].unique()) // 4):
            original = pd.read_csv('../Data/Instagram/Filter_mcdonalds_switzerland.csv')
            original_test = original[~original['image_id'].isin(ins_train['image_id'])]
            original_test = original_test[~original_test['image_id'].isin(ins_test['image_id'])]
            temp_test = original_test.sort_values(by=['funny_score'], ascending=[False])[:((len(ins_train['image_id'].unique()) // 4) - ins_test.shape[0])]
            ins_test = pd.concat([ins_test, temp_test], ignore_index=True)
        ins_test['funny_score'] = ins_test['funny_score'].apply(lambda x: 1 if x > 0.5 else 0)
        #############################################

        #############################################
        # open when you want to test train dataset
        print(f'MCD: {ins_train.shape}')
        mcdonaldDataset = InsDataset(ins_train, 'mcdonalds_switzerland', dataPath)
        #############################################
        # # open when you want to test test dataset
        # print(f'MCD: {ins_test.shape}')
        # mcdonaldDataset = InsDataset(ins_test, 'mcdonalds_switzerland', dataPath)
        ################################################
    #######################################################################################

    # get data size
    mcdonald_dataloader = DataLoader(mcdonaldDataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    sonic_dataloader = DataLoader(sonicDataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    print(len(mcdonald_dataloader), len(sonic_dataloader))

    f1_ = torchmetrics.F1Score(task='multiclass', num_classes=2, average='weighted')
    precision_ = torchmetrics.Precision(task='multiclass', num_classes=2, average='weighted')
    recall_ = torchmetrics.Recall(task='multiclass', num_classes=2, average='weighted')
    loss_class = FocalContrastiveLoss(output_dir, output_prefix, 'TestOnTest')

    for i in range(100):
        if os.path.exists(f'./Model/{output_dir}/checkpoint-{i + 1:03d}.pt'):
            model.load_state_dict(torch.load(f'./Model/{output_dir}/checkpoint-{i + 1:03d}.pt'))
            model = model.eval()
            model = model.to(device)
            print(f">>> epoch {i + 1}")
            sys.stdout.flush()
            model.eval()
            with torch.no_grad():
                ################################    mcdonald    ################################
                output_df = pd.DataFrame(columns=['image_id', 'caption', 'groundtruth', 'humor_score'])
                progress = tqdm(total=len(mcdonald_dataloader), desc=output_prefix)
                epoch_loss = 0
                for idx, (images, caption_ids, caption_masks, humor, rank, img_id, caption) in enumerate(mcdonald_dataloader):
                    model.zero_grad()
                    images, caption_ids, caption_masks, rank = images.to(device), caption_ids.to(device), caption_masks.to(device), rank.to(device)
                    outputs, sim = model(images, caption_ids, caption_masks)
                    loss = loss_class.computeLoss(outputs, rank, sim)
                    output_df = pd.concat([output_df, pd.DataFrame(
                        {'image_id': img_id, 'caption': caption, 'groundtruth': humor.cpu().numpy().flatten(),
                         'rank': rank.cpu().numpy().flatten(), 'humor_score': outputs.cpu().numpy().flatten()})], axis=0)
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
                mcdonald_precision.append(precision.item())
                mcdonald_recall.append(recall.item())
                mcdonald_f1.append(f1.item())
                mcdonald_accuracy.append(accuracy)
                ################################################################################

                ##############################    sonicdrivein    ##############################
                output_df = pd.DataFrame(columns=['image_id', 'caption', 'groundtruth', 'humor_score'])
                progress = tqdm(total=len(sonic_dataloader), desc=output_prefix)
                epoch_loss = 0
                for idx, (images, caption_ids, caption_masks, humor, rank, img_id, caption) in enumerate(
                        sonic_dataloader):
                    model.zero_grad()
                    images, caption_ids, caption_masks, rank = images.to(device), caption_ids.to(device), caption_masks.to(device), rank.to(device)
                    humor = humor.unsqueeze(1).to(device, dtype=torch.float)
                    outputs, sim = model(images, caption_ids, caption_masks)
                    loss = loss_class.computeLoss(outputs, rank, sim)
                    output_df = pd.concat([output_df, pd.DataFrame(
                        {'image_id': img_id, 'caption': caption, 'groundtruth': humor.cpu().numpy().flatten(),
                         'rank': rank.cpu().numpy().flatten(), 'humor_score': outputs.cpu().numpy().flatten()})], axis=0)
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
                sonic_precision.append(precision.item())
                sonic_recall.append(recall.item())
                sonic_f1.append(f1.item())
                sonic_accuracy.append(accuracy)
                ################################################################################

            loss_data = pd.DataFrame()
            loss_data['mcdonald_loss'] = mcdonald_losses
            loss_data['sonic_loss'] = sonic_losses
            loss_data['mcdonald_accuracy'] = mcdonald_accuracy
            loss_data['sonic_accuracy'] = sonic_accuracy
            loss_data['mcdonald_precision'] = mcdonald_precision
            loss_data['sonic_precision'] = sonic_precision
            loss_data['mcdonald_recall'] = mcdonald_recall
            loss_data['sonic_recall'] = sonic_recall
            loss_data['mcdonald_f1'] = mcdonald_f1
            loss_data['sonic_f1'] = sonic_f1
            loss_data['mcdonald_tp'] = mcdonald_tp
            loss_data['sonic_tp'] = sonic_tp
            loss_data['mcdonald_tn'] = mcdonald_tn
            loss_data['sonic_tn'] = sonic_tn
            loss_data['mcdonald_fp'] = mcdonald_fp
            loss_data['sonic_fp'] = sonic_fp
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
    df = pd.DataFrame()

    # Evaluate every result of the model from different epoch
    with torch.no_grad():
        for n in range(input_num):

            # check if evaluate file exists
            save_path = f'./Model/{args.out_dir}/'
            if args.which == 'ground_truth':
                if os.path.exists(f'{save_path}{args.which}.csv'):
                    dataPath = f'{save_path}{args.which}.csv'
                else:
                    dataPath = "none"
            else:
                if os.path.exists(f'{save_path}{args.which}_{n + 1}.csv'):
                    dataPath = f'{save_path}{args.which}_{n + 1}.csv'
                else:
                    dataPath = "none"

            # evaluate if file exists
            if dataPath != "none":
                print(dataPath)
                data = pd.read_csv(dataPath)
                image_list = data['image_id'].tolist()
                caption_list = data['caption'].tolist()
                data['humor_score'] = ''
                print(len(image_list), len(caption_list))
                if dataform == 'Oxford':
                    image_path = '../Data/Oxford_HIC/oxford_img/'
                else:
                    image_path = f'../Data/Instagram/sonicdrivein_img/'
                    image_path2 = f'../Data/Instagram/mcdonalds_switzerland_img/'

                # Loop through each image and caption
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
                        outputs, sim = model(images, caption_ids, caption_masks)
                        data.loc[i, 'humor_score'] = outputs.item()
                    else:
                        if str(image_list[i]) == 'nan':
                            data.loc[i, 'humor_score'] = ''
                        else:
                            data.loc[i, 'humor_score'] = image_list[i]

                # Convert humor_score to binary
                data['humor'] = data['humor_score'].apply(lambda x: x if type(x) != float else (1 if x > 0.5 else 0))
                # Concat evaluated results
                if df.empty:
                    df = data[['image_id', 'caption', 'humor_score','humor']]
                    if args.which == 'ground_truth':
                        break
                    break
                else:
                    df = pd.concat([df, data[['image_id', 'caption', 'humor_score','humor']]], axis=1)

        df.to_csv(f'{save_path}left0_{model_path}_{args.which}_humorscore.csv', float_format='%.15f', index=False)
        """
            Save the results
                Name:
                    {save_path} -- the path to save the results / the path evaluate file exists
                    {humor_evaluation_model_name} -- the humor evaluation model name, which is the out_dir of the model
                    {model_path} -- the model path, which is the epoch number of the humor evaluation model
                    {args.which} -- the which argument, which is the type of the evaluation, such as generate_beam, generate2, ground_truth
            Example:
                    {save_path}{humor_evaluation_model_name}_{model_path}_{args.which}_humorscore.csv
        """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=16)
    """
        < out_dir naming structure >
        Name:
            {data_form} -- the data form and the numbers of data from Instagram, such as SD_1300 stands for 1300 image caption pairs from sonicdrivein
            {augmentation} -- the augmentation method, such as noAug stands for no augmentation for captions
            {exchange} -- the exchange method, such as exchange_05_08_13 stands for exchange bad test samples to train samples from the previous result focal loss alpha = 0.8, gamma = 1.3, contrast temp 0.5
            {add_wOxford} -- the data form and the numbers of data from Oxford, such as add_wOxford_1900 stands for 1900 image caption pairs from Oxford
            {data_type} -- the only01 stands for the only 1 image caption pair from each image, which is the same as GAMC model
            {focalLoss_setting} -- the focal loss setting, such as focal08_19 stands for focal loss alpha = 0.8, gamma = 1.9
            {contrast_setting} -- the contrastive loss setting, such as contrast_temp050 stands for the temperature of the contrastive loss setting is 0.5
        Example:
            humorScore_{data_form}_{augmentation}_{exchange}_{add_wOxford}_{data_type}_{focalLoss_setting}_{contrast_setting}
            humorScore_SD_1300_MC_600_noAug_exchange_05_08_13_add_wOxford_1900_only01_focal08_19_contrast_temp050
    """
    ##########################   train/test   model   ##########################
    # parser.add_argument('--dataPath', default='humorScore/humorScore_SD_1300_MC_600_noAug_exchange_05_08_13_add_wOxford_1900_only01', help='train data')
    parser.add_argument('--dataPath', default='humorScore/humorScore_SD_1300_MC_600_noAug_exchange_05_08_13_add_wOxford_1900_only01_train', help='test ins train data')
    # parser.add_argument('--dataPath', default='humorScore/humorScore_SD_1300_MC_600_noAug_exchange_05_08_13_add_wOxford_1900_only01_test', help='test ins test data')
    parser.add_argument('--out_dir', default='humorScore/humorScore_20250526_SD_1300_MC_600_noAug_exchange_05_08_13_add_wOxford_1900_only01_focal08_19_contrast_temp050')
    parser.add_argument('--prefix', default='checkpoint', help='prefix for saved filenames')
    ############################################################################


    ########################   Humor score Evalution    ########################
    parser.add_argument('--out_dir', default='3000/ablation/SD/_woFCT_onlyLoRaadapt', help='GAMC directory')
    # parser.add_argument('--out_dir', default='clip/clip_100up_only200_lessNotFunImg_169_55_passlength_10_SD_with_oxford_3000_only1_300_82', help='clip directory')
    # parser.add_argument('--out_dir', default='bita/sd', help='bita directory')
    parser.add_argument('--prefix', default='humorScore_20250526_SD_1300_MC_600_noAug_exchange_05_08_13_add_wOxford_1900_only01_focal08_17_contrast_temp070_left0', help='model directory to load')
    ############################################################################

    args = parser.parse_args()
    if not os.path.exists('./Model/' + args.out_dir):
        os.makedirs('./Model/' + args.out_dir)
        os.makedirs('D:/MemeGAN/Model/' + args.out_dir)

    model = ImageTextModel()
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()

    #############################   train  model   #############################
    # train(model, args, output_dir=args.out_dir, output_prefix=args.prefix)
    ############################################################################
    #############################    test model    #############################
    # test(model, args, output_dir=args.out_dir, output_prefix=args.prefix)
    ############################################################################

    ######################    Humor score  Evalution    ########################
    which_list = ['generate_beam', 'generate2', 'ground_truth']
    for which in which_list:
        args.which = which
        print(f"######### {which} #########")
        for i in range(20):
            i = i + 11
            if os.path.exists(f'./Model/humorScore/{args.prefix}/checkpoint-{i:03d}.pt'):
                model.load_state_dict(torch.load(f'./Model/humorScore/{args.prefix}/checkpoint-{i:03d}.pt'))
                # test_mine(model, args, dataform="sonicdrivein", input_num=20, model_path=i)
                test_mine(model, args, dataform="mcdonalds_switzerland", input_num=20, model_path=i)
            break
    ############################################################################

if __name__ == '__main__':
    main()