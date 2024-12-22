import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, Gemma2ForCausalLM
eps = torch.finfo(torch.bfloat16).eps
from typing import Tuple, Optional, Union
import numpy as np
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup
# eps = torch.finfo(torch.bfloat16).eps
eps = 1e-5
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
class OxfordDataset(torch.utils.data.Dataset):
    def __init__(self, text, image, funny_score):
        self.text = text
        self.image = image
        self.funny_score = funny_score

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        imageData = torch.load('../../Oxford_HIC/ImageClip/'+ self.image[idx] +'.pt', weights_only=False)
        # all dtype to torch.float16
        imageData = imageData

        return self.text[idx], imageData, self.funny_score[idx]


def train():
    epochs = 200
    batch_size = 5
    optimizer_Former_lr = 1e-5
    save_name = '20241221_Clip_Clip_Clip_noGemma_prefix10_mask'
    if not os.path.exists('./Model/' + save_name):
        os.makedirs('./Model/' + save_name)
        os.makedirs('D:/MemeGAN/Model/' + save_name)
    ################ right ################
    # twoModel = False
    # checkpoint_folder = '20241201_wo_coAttention_temp'
    # checkpoint_Former = 'D:/MemeGAN/Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetFormer_83.pth'
    #######################################
    # twoModel = True
    # checkpoint_folder = '20241204_noprompt_base_20241201_coAttention_temp'
    # checkpoint_Former = './Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetFormer_46.pth'
    # checkpoint_Generator = './Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetLLM_46.pth'
    #######################################
    # if args.img - dir == 'Oxford_HIC':
    dirPath = '../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
    # else:
    # dirPath = '../Data/Instagram/Filter_' + 'wendys' + '.csv'
    # imgPath = '../Data/Instagram/' + 'wendys' + '_img/'
    # load data
    data = pd.read_csv(dirPath)
    print("shape of data: ", data.shape)
    data = data.sample(n=10000, random_state=42, replace=True).reset_index(drop=True)
    # frac = 0.05 ==> 5% of the data = 169904
    # n = 169920 ==> 72 * 2360 = 169920 (F2G)
    # n = 169988 ==> 91 * 1868 = 169988 (G2F)
    print("sample of data: ", data.shape)

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train_text = train['caption'].tolist()
    train_image = train['image_id'].tolist()
    train_funny_score = train['funny_score'].tolist()
    test_text = test['caption'].tolist()
    test_image = test['image_id'].tolist()
    test_funny_score = test['funny_score'].tolist()

    train_dataset = OxfordDataset(train_text, train_image, train_funny_score)
    test_dataset = OxfordDataset(test_text, test_image, test_funny_score)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)

    ### GPT2 #########################################################################################
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer.pad_token = tokenizer.eos_token
    # gpt = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    # embedding_size = gpt.transformer.wte.weight.shape[1]
    # print('embedding_size: ', embedding_size)
    # prefix = image embedding, token = text embedding
    ########################################################################################################
    ### 官方的Gemma #########################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2b = 2304, 9b = 3584, 27b = 4608
    embedding_size = 2304
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    # gemmaConfig = AutoConfig.from_pretrained('google/gemma-2-2b-it')
    gemma = Gemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16)

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
    # a = count_trainable_parameters(gemma)
    # gemma = get_peft_model(gemma, LORAconfig)
    # b = count_trainable_parameters(gemma)
    # #留下小數點後兩位就好
    # percent = round((b / a) * 100, 3)
    # print("Before: ", a, "After: ", b, "Percent: ", percent, "%")
    ########################################################################################################

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
        def __init__(self, h_dim):
            super(MlpTransformer, self).__init__()
            self.fc1 = nn.Linear(gpt_embedding_size, h_dim)
            self.relu = nn.functional.relu
            self.fc2 = nn.Linear(h_dim, gpt_embedding_size)
            self.dropout = nn.Dropout(0.0)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x

    class MultiHeadAttention(nn.Module):

        def __init__(self, num_heads, bias=True, dropout=0.):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            head_dim = gpt_embedding_size // num_heads
            self.scale = head_dim ** -0.5
            self.to_queries = nn.Linear(gpt_embedding_size, gpt_embedding_size, bias=bias)
            self.to_keys_values = nn.Linear(gpt_embedding_size, gpt_embedding_size * 2, bias=bias)
            self.project = nn.Linear(gpt_embedding_size, gpt_embedding_size)
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
        def __init__(self,  mlp_ratio=4., bias=False, dropout=0.):
            super(TransformerLayer, self).__init__()
            self.norm1 = nn.LayerNorm(gpt_embedding_size, eps=eps)
            self.attn = MultiHeadAttention(8, bias=bias, dropout=dropout)
            self.norm2 = nn.LayerNorm(gpt_embedding_size, eps=eps)
            self.mlp = MlpTransformer(int(gpt_embedding_size * mlp_ratio))

        def forward_with_attention(self, x, y=None, mask=None):
            x_, attention = self.attn(self.norm1(x), y, mask)
            x = x + x_
            x = x + self.mlp(self.norm2(x))
            return x, attention

        def forward(self, x, y=None, mask=None):
            x = x + self.attn(self.norm1(x), y, mask)[0]
            x = x + self.mlp(self.norm2(x))
            return x

    class Transformer(nn.Module):
        def __init__(self, num_layers: int, mlp_ratio: float = 2., enc_dec: bool = False):
            super(Transformer, self).__init__()
            self.enc_dec = enc_dec
            if enc_dec:
                num_layers = num_layers * 2
            layers = []
            for i in range(num_layers):
                if i % 2 == 0 and enc_dec:  # cross
                    layers.append(TransformerLayer(mlp_ratio))
                elif enc_dec:  # self
                    layers.append(TransformerLayer(mlp_ratio))
                else:  # self or cross
                    layers.append(TransformerLayer(mlp_ratio))
            self.layers = nn.ModuleList(layers)

        def forward_with_attention(self, x, y=None, mask=None):
            attentions = []
            for layer in self.layers:
                x, att = layer.forward_with_attention(x, y, mask)
                attentions.append(att)
            return x, attentions

        def forward(self, x, y=None, mask=None):
            for i, layer in enumerate(self.layers):
                x = layer(x, y, mask)
            return x

    class TransformerMapper(nn.Module):
        def __init__(self):
            super(TransformerMapper, self).__init__()
            self.transformer = Transformer(num_layers=8, enc_dec=False)
            self.linear = nn.Linear(512, 10*gpt_embedding_size)
            self.prefix_const = nn.Parameter(torch.randn(10, gpt_embedding_size), requires_grad=True)

        def forward(self, x):
            x = self.linear(x).view(x.shape[0], 10, gpt_embedding_size)
            prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], -1, -1).to(device).to(torch.bfloat16)
            prefix = torch.cat((x, prefix), dim=1)
            out = self.transformer(prefix)[:, 10:]
            return out

    class Generator(nn.Module):
        def __init__(self, Former, gemma, depth=12):
            super(Generator, self).__init__()
            if Former == "MLP":
                self.Former = MLP()
            else:
                self.Former = TransformerMapper()
            self.gemma = gemma
            self.gemma.eval()
            for param in self.gemma.parameters():
                param.requires_grad = False

        def forward(self, image, text_id, mask, mode):
            if mode == 'test':
                dummy = torch.full((image.shape[0], 64), 0, dtype=torch.long).to(device)
                text_embedd = self.gemma.model.embed_tokens(dummy)
            else:
                text_embedd = self.gemma.model.embed_tokens(text_id)
            ##############################################   generate ##############################################
            image_G = self.Former(image)
            embedding_cat = torch.cat((image_G, text_embedd), dim=1)
            ########################################### feature fusion ###########################################
            gemmaOutput = self.gemma(inputs_embeds=embedding_cat, labels=labels, return_dict = True)
            logits = gemmaOutput.logits

            return logits

    # NetFormer = Former().to(torch.bfloat16).to(device)
    Generator = Generator(Former= "Trans").to(torch.bfloat16).to(device)
    optimizer_Former = optim.Adam(Generator.parameters(), lr= optimizer_Former_lr)
    counter = 0
    for param in Generator.parameters():
        if param.requires_grad:
            counter += param.numel()
    print("number of parameters: ", counter)
    scheduler = get_linear_schedule_with_warmup(
        optimizer_Former, num_warmup_steps=5000, num_training_steps=epochs * len(train_loader)
    )
    train_losses_Former = []
    test_losses_Former = []
    save = []
    present_epoch = 1
    best_train_loss_Former = 9999999999
    best_test_loss_Former = 9999999999
    gemma_loss_list = []
    fc_loss_list = []

    # checkpoint_Former = torch.load(checkpoint_Former)
    # Generator.Former.load_state_dict(checkpoint_Former['model_state_dict'])
    #
    # if twoModel:
    #     checkpoint_Generator = torch.load(0)
    #     Generator.load_state_dict(checkpoint_Generator['model_state_dict'])
    #     present_epoch = checkpoint_Former['epoch'] + 1
    #     del checkpoint_Generator
    #
    # del checkpoint_Former
    # gc.collect()

    def loss_function(logits, labels):
        logits = logits[:, 9:-1]
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        loss = nn.CrossEntropyLoss(ignore_index=0)(logits, labels)
        return loss

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        print("---------------------------------------- epoch " + str(
            epoch + present_epoch) + " ---------------------------------------")
        train_loss_Former = 0
        test_loss_Former = 0
        ###################################### Train ######################################
        with tqdm(train_loader, unit="batch", leave=True) as tepoch:
            Generator.train()
            for idx, (text, image, funny_score) in enumerate(tepoch):
                Generator.zero_grad()
                text_data = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
                text_id = text_data['input_ids'].to(device)
                mask = torch.cat ((torch.ones(text_id.shape[0], 10), text_data['attention_mask']), dim=1).to(device)
                image = image.to(device, dtype=torch.bfloat16)
                # funny_score = funny_score.to(device, dtype=torch.bfloat16)
                logits = Generator(image, text_id, mask, mode='train')
                loss = loss_function(logits, text_id)
                loss.backward()
                optimizer_Former.step()
                scheduler.step()
                optimizer_Former.zero_grad()
                train_loss_Former += loss.item()
                tepoch.set_postfix(loss=train_loss_Former / (idx + 1))
                ##########################################################################
        train_loss_Former /= len(train_loader)
        train_losses_Former.append(train_loss_Former)
        ##################################### Test ######################################
        with tqdm(test_loader, unit="batch", leave=True) as tepoch:
            Generator.eval()
            with torch.no_grad():
                for idx, (text, image, funny_score) in enumerate(tepoch):
                    text_data = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
                    text_id = text_data['input_ids'].to(device)
                    mask = torch.cat((torch.ones(text_id.shape[0], 10), text_data['attention_mask']), dim=1).to(device)
                    image = image.to(device, dtype=torch.bfloat16)
                    # funny_score = funny_score.to(device, dtype=torch.bfloat16)
                    logits = Generator(image, text_id , mask, mode='test')
                    loss = loss_function(logits, text_id)
                    test_loss_Former += loss.item()
                    tepoch.set_postfix(loss=test_loss_Former / (idx + 1))
        test_loss_Former /= len(test_loader)
        test_losses_Former.append(test_loss_Former)
        ###################################### Save ######################################

        if test_loss_Former < best_test_loss_Former and train_loss_Former < best_train_loss_Former:
            best_test_loss_Former = test_loss_Former
            best_train_loss_Former = train_loss_Former
            torch.save({
                'epoch': epoch + present_epoch,
                'model_state_dict': Generator.state_dict(),
                'optimizer_state_dict': optimizer_Former.state_dict(),
                'loss': loss,
            }, './Model/' + save_name + '/' + save_name + '_NetLLM_' + str(epoch + present_epoch) + '.pth')
            save.append("V")

        else:
            save.append(" ")

        loss_data = pd.DataFrame()
        loss_data['train_loss'] = train_losses_Former
        loss_data['test_loss'] = test_losses_Former
        loss_data['save'] = save
        loss_data.to_csv('./Model/' + save_name + '/' + save_name + '_loss.csv', index=False)

        plt.plot(train_losses_Former, label='train')
        plt.plot(test_losses_Former, label='test')
        plt.legend()
        plt.savefig('./Model/' + save_name + '/' + save_name + '_loss.png')
        plt.show()

if __name__ == '__main__':
    train()