# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers import AutoConfig, AutoTokenizer, Gemma2ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import PIL.Image
from typing import Tuple, Optional, Union, Any
from tqdm import tqdm
import os
import pickle
import sys
from typing import Tuple, Optional, Union, Any
from typing import Tuple, List, Union, Optional
import numpy as np
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
from nltk.translate.bleu_score import sentence_bleu
import gc
import loralib as lora
from Citations.Parrot_Paraphraser.parrot.filters import Fluency
from scipy.special import softmax
import clip
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    def __init__(self, prefix_length, cp_num, train_caption, test_caption, train_image_id_list, test_image_id_list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.prefix_length = prefix_length
        self.cp_num = cp_num
        self.train_caption = train_caption
        self.test_caption = test_caption
        self.train_image_id_list = train_image_id_list
        self.test_image_id_list = test_image_id_list
        print("parrot model loading")
        self.fluency_score  = Fluency()
        self.fluency_score.fluency_model = self.fluency_score.fluency_model.to(self.device)
        print("parrot model loaded")
        self.clip, self.clip_preprocess = clip.load("ViT-B/32")
        self.clip.to(device)
        self.clip.eval()
        print("clip model loaded")

    def fitCounter(self, data_mode, caption, dataIndex):
        bleu1_sum = 0
        bleu2_sum = 0
        bleu3_sum = 0
        bleu4_sum = 0
        if data_mode == "test":
            fitCount = 0
            caption_words = caption.split()
            index = self.test_image_id_list[dataIndex]
            for j in range(len(self.test_caption[index])):
                bleu1_sum += sentence_bleu([self.test_caption[index][j].split()], caption_words, weights=(1, 0, 0, 0))
                bleu2_sum += sentence_bleu([self.test_caption[index][j].split()], caption_words, weights=(0, 1, 0, 0))
                bleu3_sum += sentence_bleu([self.test_caption[index][j].split()], caption_words, weights=(0, 0, 1, 0))
                bleu4_sum += sentence_bleu([self.test_caption[index][j].split()], caption_words, weights=(0, 0, 0, 1))
                gt_words = self.test_caption[index][j].split()
                for k in range(len(caption_words)):
                    if caption_words[k] in gt_words:
                        fitCount += 1
            gtNum = len(self.test_caption[index])
        else:
            fitCount = 0
            caption_words = caption.split()
            index = self.train_image_id_list[dataIndex]
            for j in range(len(self.train_caption[index])):
                bleu1_sum += sentence_bleu([self.train_caption[index][j].split()], caption_words, weights=(1, 0, 0, 0))
                bleu2_sum += sentence_bleu([self.train_caption[index][j].split()], caption_words, weights=(0, 1, 0, 0))
                bleu3_sum += sentence_bleu([self.train_caption[index][j].split()], caption_words, weights=(0, 0, 1, 0))
                bleu4_sum += sentence_bleu([self.train_caption[index][j].split()], caption_words, weights=(0, 0, 0, 1))
                gt_words = self.train_caption[index][j].split()
                for k in range(len(caption_words)):
                    if caption_words[k] in gt_words:
                        fitCount += 1
            gtNum = len(self.train_caption[index])
        bleu1_sum /= gtNum
        bleu2_sum /= gtNum
        bleu3_sum /= gtNum
        bleu4_sum /= gtNum
        return fitCount, gtNum, bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum

    def parrot_fluency(self, caption):
        input_ids = self.fluency_score.fluency_tokenizer("Sentence: " + caption, return_tensors='pt', truncation=True)
        input_ids = input_ids.to(device)
        prediction = self.fluency_score.fluency_model(**input_ids)
        scores = prediction[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        fluency_score = scores[1]  # LABEL_0 = Bad Fluency, LABEL_1 = Good Fluency
        return fluency_score

    def diversity_score(self, captionList):
        # tokenize  and extract text features
        text_tokens = clip.tokenize(captionList, context_length=77, truncate=True).to(device)
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

    def predict(self, data_mode,tokens, masks, prefixs, text_gt, model, emotion, sentiment, humor, funnyscore):
        self.embedding_size = model.embedding_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens, masks, prefixs = tokens.to(device), masks.to(device), prefixs.to(device, dtype=torch.bfloat16)
        emotion, sentiment, humor = emotion.to(device), sentiment.to(device), humor.to(device)
        # emotion, sentiment, humor = emotion.to(device, dtype=torch.bfloat16), sentiment.to(device, dtype=torch.bfloat16), humor.to(device, dtype=torch.bfloat16)

        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-1B-Base")
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        model.to(device, dtype=torch.bfloat16)
        for i, text in enumerate(text_gt):
            print('====================== '+data_mode+' '+ str(i+1) +'======================')
            print('--------- ground truth ---------')
            print(text)
            print('--------- generate_beam ---------')
            if model.LoRaActivated:
                embedding_emotion = model.falcon.base_model.model.model.embed_tokens(emotion[i].unsqueeze(0))
                embedding_sentiment = model.falcon.base_model.model.model.embed_tokens(sentiment[i].unsqueeze(0))
                embedding_humor = model.falcon.base_model.model.model.embed_tokens(humor[i].unsqueeze(0))
            else:
                embedding_emotion = model.falcon.model.embed_tokens(emotion[i].unsqueeze(0))
                embedding_sentiment = model.falcon.model.embed_tokens(sentiment[i].unsqueeze(0))
                embedding_humor = model.falcon.model.embed_tokens(humor[i].unsqueeze(0))

            empty_ESH = torch.zeros(embedding_emotion.shape[0], 1, embedding_emotion.shape[2], dtype=torch.bfloat16, device=embedding_emotion.device)
            embedding_ESH = torch.cat((empty_ESH, embedding_emotion, embedding_sentiment, embedding_humor), dim=1)

            # emotion_proj = model.emotion_linear(emotion[i].unsqueeze(0)).unsqueeze(1)
            # sentiment_proj = model.sentiment_linear(sentiment[i].unsqueeze(0)).unsqueeze(1)
            # humor_proj = model.humor_linear(humor[i].unsqueeze(0)).unsqueeze(1)
            # embedding_ESH = torch.cat((emotion_proj, sentiment_proj, humor_proj), dim=1)
            # embedding_ESH = model.ESH_linear(embedding_ESH.transpose(1, 2)).transpose(1, 2)
            ############################################ 202503 ##############################################
            clip_projections = model.clip_project(prefixs[i].unsqueeze(0)).view(-1, model.prefix_length, model.embedding_size)
            visual_projections_swin = model.visual_project_swin(embedding_ESH, clip_projections)
            visual_projections_ESH = model.visual_project_ESH(clip_projections, embedding_ESH)
            ########################################################################################
            ## 20250302_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_swin_tf8  ###
            ########################################################################################
            prefix_embed = torch.cat((visual_projections_swin, visual_projections_ESH), dim=1)
            prefix_embed = model.visual_project(prefix_embed.transpose(1, 2)).transpose(1, 2)
            ########################################################################################
            #### 20250303_oxford_lower_800up_only800_rest_300up_top300_ESH_co_concat_swin_tf8   ####
            ########################################################################################
            # prefix_embed = visual_projections_swin
            #######################################################################################
            # 20250310_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_add_768_swin_tf8    #
            # 20250307_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_768_swin_tf8 #
            #######################################################################################
            # prefix_embed = model.linear(prefix_embed)
            ################################################################################

            caption = generate_beam(model, tokenizer, embed=prefix_embed)[0]
            caption_token = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=74)

            print(caption)
            if self.generate_beam_output != None:
                self.generate_beam_output = torch.cat((self.generate_beam_output, caption_token['input_ids']), dim=0)
            else:
                self.generate_beam_output = caption_token['input_ids']
            self.generate_beam_text.append(caption)
            self.generate_beam_fluency.append(self.parrot_fluency(caption))
            fitCount, gtNum, bleu1, bleu2, bleu3, bleu4 = self.fitCounter(data_mode, caption, i)
            self.generate_beam_fitCount.append(fitCount)
            self.generate_beam_gtNum.append(gtNum)
            self.generate_beam_bleu1.append(bleu1)
            self.generate_beam_bleu2.append(bleu2)
            self.generate_beam_bleu3.append(bleu3)
            self.generate_beam_bleu4.append(bleu4)
            print('---------generate2---------')
            caption = generate2(model, tokenizer, embed=prefix_embed)
            caption_token = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=74)
            print(caption)
            if self.generate2_output != None:
                self.generate2_output = torch.cat((self.generate2_output, caption_token['input_ids']), dim=0)
            else:
                self.generate2_output = caption_token['input_ids']
            self.generate2_text.append(caption)
            self.generate2_fluency.append(self.parrot_fluency(caption))
            fitCount, gtNum, bleu1, bleu2, bleu3, bleu4 = self.fitCounter(data_mode, caption, i)
            self.generate2_fitCount.append(fitCount)
            self.generate2_gtNum.append(gtNum)
            self.generate2_bleu1.append(bleu1)
            self.generate2_bleu2.append(bleu2)
            self.generate2_bleu3.append(bleu3)
            self.generate2_bleu4.append(bleu4)
            print('---------train---------')
            # embedding_text = model.gemma.model.embed_tokens(tokens[i].unsqueeze(0))
            # embedding_text = model.gemma.base_model.model.model.embed_tokens(tokens[i].unsqueeze(0))
            # embedding_text = model.gpt.transformer.wte(tokens[i].unsqueeze(0))
            if model.LoRaActivated:
                embedding_text = model.falcon.base_model.model.model.embed_tokens(tokens[i].unsqueeze(0))
            else:
                embedding_text = model.falcon.model.embed_tokens(tokens[i].unsqueeze(0))
            print(prefix_embed.shape, embedding_text.shape)
            embedding_cat = torch.cat((prefix_embed, embedding_text), dim=1)
            # out = model.gemma(inputs_embeds=embedding_cat, attention_mask=masks[i].unsqueeze(0))
            # out = model.gpt(inputs_embeds=embedding_cat, attention_mask=masks[i].unsqueeze(0))
            out = model.falcon(inputs_embeds=embedding_cat, attention_mask=masks[i].unsqueeze(0))
            logits = out.logits[:, self.prefix_length-1: -1]
            if model.mode == 'lora' or model.LoRaActivated:
                output_fc = model.funnyscore_mlp1(embedding_cat).squeeze(-1)
                output_fc = model.funnyscore_relu(output_fc)
                output_fc = model.funnyscore_mlp2(output_fc).squeeze(-1)
                output_fc = model.funnyscore_sigmoid(output_fc)
                capLoss, fcLoss, loss = combine_loss(logits, tokens[i], funnyscore[i].unsqueeze(0).to(device, dtype=torch.bfloat16), output_fc)
                self.train_caption_loss.append(capLoss.item())
                self.train_fc_loss.append(fcLoss.item())
            else:
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[i].flatten(), ignore_index=0)
                # loss = PCloss(logits, tokens[i].unsqueeze(0))
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
            self.train_fluency.append(self.parrot_fluency(caption))
            fitCount, gtNum, bleu1, bleu2, bleu3, bleu4 = self.fitCounter(data_mode, caption, i)
            self.train_fitCount.append(fitCount)
            self.train_gtNum.append(gtNum)
            self.train_bleu1.append(bleu1)
            self.train_bleu2.append(bleu2)
            self.train_bleu3.append(bleu3)
            self.train_bleu4.append(bleu4)

        if data_mode == "train":
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
            test['Name'] = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9','test10',
                            'train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7',
                            'train8', 'train9', 'train10']
            test['image_id'] = self.test_image_id_list[:10] + self.train_image_id_list[:10]
            generate_beam_df = pd.DataFrame(self.generate_beam_output.cpu().detach(),columns=[f"{i}" for i in range(0, 74)])
            generate_beam_df = pd.concat([test, generate_beam_df, dataframe_Name("-", 74, 16)], axis=1)
            generate_beam_df['text'] = self.generate_beam_text
            generate_beam_df['fitCount'] = self.generate_beam_fitCount
            generate_beam_df['gtNum'] = self.generate_beam_gtNum
            generate_beam_df['fluency'] = self.generate_beam_fluency
            generate_beam_df['diversity'] = self.diversity_score(self.generate_beam_text)
            generate_beam_df['bleu1'] = self.generate_beam_bleu1
            generate_beam_df['bleu2'] = self.generate_beam_bleu2
            generate_beam_df['bleu3'] = self.generate_beam_bleu3
            generate_beam_df['bleu4'] = self.generate_beam_bleu4
            test_avgscore = pd.DataFrame()
            train_avgscore = pd.DataFrame()
            for column in generate_beam_df.columns:
                if column == 'Name':
                    train_avgscore[column] = ["train_avg"]
                    test_avgscore[column] = ["test_avg"]
                elif pd.api.types.is_numeric_dtype(generate_beam_df[column]):
                    train_avgscore[column] = [generate_beam_df[column][10:].mean()]
                    test_avgscore[column] = [generate_beam_df[column][:10].mean()]
                else:
                    train_avgscore[column] = ["-"]
                    test_avgscore[column] = ["-"]
            generate_beam_df = pd.concat([generate_beam_df, test_avgscore, train_avgscore], axis=0)

            generate2_df = pd.DataFrame(self.generate2_output.cpu().detach(), columns=[f"{i}" for i in range(0, 74)])
            generate2_df = pd.concat([test, generate2_df, dataframe_Name("-", 74, 16)], axis=1)
            generate2_df['text'] = self.generate2_text
            generate2_df['fitCount'] = self.generate2_fitCount
            generate2_df['gtNum'] = self.generate2_gtNum
            generate2_df['fluency'] = self.generate2_fluency
            generate2_df['diversity'] = self.diversity_score(self.generate2_text)
            generate2_df['bleu1'] = self.generate2_bleu1
            generate2_df['bleu2'] = self.generate2_bleu2
            generate2_df['bleu3'] = self.generate2_bleu3
            generate2_df['bleu4'] = self.generate2_bleu4
            test_avgscore = pd.DataFrame()
            train_avgscore = pd.DataFrame()
            for column in generate2_df.columns:
                if column == 'Name':
                    train_avgscore[column] = ["train_avg"]
                    test_avgscore[column] = ["test_avg"]
                elif pd.api.types.is_numeric_dtype(generate2_df[column]):
                    train_avgscore[column] = [generate2_df[column][10:].mean()]
                    test_avgscore[column] = [generate2_df[column][:10].mean()]
                else:
                    train_avgscore[column] = ["-"]
                    test_avgscore[column] = ["-"]
            generate2_df = pd.concat([generate2_df, test_avgscore, train_avgscore], axis=0)

            train_df = pd.DataFrame(self.train_output.cpu().detach(), columns=[f"{i}" for i in range(0, 489)])
            train_df = pd.concat([test, train_df], axis=1)
            train_df['loss'] = self.train_loss
            # if model.mode == 'lora' or model.LoRaActivated:
            #     train_df['caption_loss'] = self.train_caption_loss
            #     train_df['fc_loss'] = self.train_fc_loss
            train_df['text'] = self.train_text
            train_df['fitCount'] = self.train_fitCount
            train_df['gtNum'] = self.train_gtNum
            train_df['fluency'] = self.train_fluency
            train_df['diversity'] = self.diversity_score(self.train_text)
            train_df['bleu1'] = self.train_bleu1
            train_df['bleu2'] = self.train_bleu2
            train_df['bleu3'] = self.train_bleu3
            train_df['bleu4'] = self.train_bleu4
            test_avgscore = pd.DataFrame()
            train_avgscore = pd.DataFrame()
            for column in train_df.columns:
                if column == 'Name':
                    train_avgscore[column] = ["train_avg"]
                    test_avgscore[column] = ["test_avg"]
                elif pd.api.types.is_numeric_dtype(train_df[column]):
                    train_avgscore[column] = [train_df[column][10:].mean()]
                    test_avgscore[column] = [train_df[column][:10].mean()]
                else:
                    train_avgscore[column] = ["-"]
                    test_avgscore[column] = ["-"]
            train_df = pd.concat([train_df, test_avgscore, train_avgscore], axis=0)

            print(dataframe_Name("generate_beam").shape, generate_beam_df.shape, dataframe_Name("generate2").shape,
                  generate2_df.shape, dataframe_Name("train").shape, train_df.shape)
            final = pd.concat([dataframe_Name("generate_beam"), generate_beam_df, dataframe_Name("generate2"),
                               generate2_df, dataframe_Name("train"), train_df], axis=0)
            print(final.shape)
            print(final.columns)
            # final.to_csv(f'./Model/{save_file}/test_{self.cp_num:03d}.csv', float_format='%.15f', index=False)

            # final.to_csv(f'./Model/{save_file}/{save_file}_test_{self.cp_num:03d}.csv', float_format='%.15f', index=False)
            final.to_csv(f'./Model/{save_file}/test_{self.cp_num:03d}.csv', float_format='%.15f', index=False)
            # final.to_csv(f'./Model/{save_file}/Generate_mcdonalds_all.csv', index=False)

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

    def __init__(self, mode: str, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=True, dropout=0., act=nnf.relu,
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
        ### clip ###
        # x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        ### swin ###
        if x.shape[2] == 768:
            x = self.linear(x)
        ############
        ### 768 ###
        # if x.shape[2] != 768:
        #     x = self.linear(x)
        ############
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, mode: str, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int,
                 num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(mode, dim_embedding, 8, num_layers)
        ### clip ###
        # self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        ### swin ###
        self.linear = nn.Linear(768, dim_embedding)
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

    def __init__(self, mode: str, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int,
                 num_layers: int = 8):
        super(CrossTransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(mode, dim_embedding, 8, num_layers, cross=True)
        ### clip ###
        # self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        ### swin ###
        self.linear = nn.Linear(768, dim_embedding)
        ############
        ### 768 ###
        # self.linear = nn.Linear(2048, dim_embedding)
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
        if self.LoRaActivated:
            embedding_text = self.falcon.base_model.model.model.embed_tokens(tokens)
            embedding_emotion = self.falcon.base_model.model.model.embed_tokens(emotion)
            embedding_sentiment = self.falcon.base_model.model.model.embed_tokens(sentiment)
            embedding_humor = self.falcon.base_model.model.model.embed_tokens(humor)
        else:
            embedding_text = self.falcon.model.embed_tokens(tokens)
            embedding_emotion = self.falcon.model.embed_tokens(emotion)
            embedding_sentiment = self.falcon.model.embed_tokens(sentiment)
            embedding_humor = self.falcon.model.embed_tokens(humor)
        empty_ESH = torch.zeros(embedding_emotion.shape[0], 1, embedding_emotion.shape[2], dtype=torch.bfloat16,
                                device=embedding_text.device)
        embedding_ESH = torch.cat((empty_ESH, embedding_emotion, embedding_sentiment, embedding_humor), dim=1)
        # emotion_proj = self.emotion_linear(emotion).unsqueeze(1)
        # sentiment_proj = self.sentiment_linear(sentiment).unsqueeze(1)
        # humor_proj = self.humor_linear(humor).unsqueeze(1)
        # embedding_ESH = torch.cat((emotion_proj, sentiment_proj, humor_proj), dim=1)
        # embedding_ESH = self.ESH_linear(embedding_ESH.transpose(1, 2)).transpose(1, 2)
        ############################################ 202503 ##############################################
        clip_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.embedding_size)
        visual_projections_swin = self.visual_project_swin(embedding_ESH, clip_projections)
        visual_projections_ESH = self.visual_project_ESH(clip_projections, embedding_ESH)
        ########################################################################################
        ### 20250302_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_swin_tf8  ###
        ########################################################################################
        visual_projections = torch.cat((visual_projections_swin, visual_projections_ESH), dim=1)
        visual_projections = self.visual_project(visual_projections.transpose(1, 2)).transpose(1, 2)
        ########################################################################################
        #######  20250303_oxford_lower_800up_only800_rest_300up_top300_ESH_co_swin_tf8   #######
        ########################################################################################
        # visual_projections = visual_projections_swin
        #######################################################################################
        # 20250310_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_add_768_swin_tf8    #
        # 20250307_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_768_swin_tf8 #
        #######################################################################################
        # visual_projections = self.linear(visual_projections)
        #######################################################################################
        embedding_cat = torch.cat((visual_projections, embedding_text), dim=1)
        ##################################################################################################

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        # out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        # out = self.gemma(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        out = self.falcon(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        if self.mode == 'lora' or self.LoRaActivated == True:
            fc = self.funnyscore_mlp1(embedding_cat).squeeze(-1)
            fc = self.funnyscore_relu(fc)
            fc = self.funnyscore_mlp2(fc).squeeze(-1)
            fc = self.funnyscore_sigmoid(fc)
            return out, fc
        else:
            return out

    def __init__(self, mode: str, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.LoRaActivated = False
        self.mode = mode
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
        self.embedding_size = self.falcon.model.embed_tokens.weight.shape[1]
        # self.embedding_size = 768
        # self.falcon.eval()
        # for param in self.falcon.parameters():
        #     param.requires_grad = False
        self.clip_project = TransformerMapper(mode, prefix_size, self.embedding_size, prefix_length, clip_length, num_layers)
        self.visual_project_swin = CrossTransformerMapper(mode, prefix_size, self.embedding_size, prefix_length, clip_length, num_layers)
        self.visual_project_ESH = CrossTransformerMapper(mode, prefix_size, self.embedding_size, prefix_length, clip_length, num_layers)
        #######################################################################################
        ### 20250228_oxford_lower_800up_only800_rest_300up_top300_ESH_cross_concat_swin_tf8 ###
        #######################################################################################
        self.visual_project = nn.Linear(128, 64)
        self.linear = nn.Linear(self.embedding_size, 2048)

        self.emotion_linear = nn.Linear(384, 768)
        self.sentiment_linear = nn.Linear(384, 768)
        self.humor_linear = nn.Linear(768, 768)
        self.ESH_linear = nn.Linear(3, 64)
        if mode == 'lora':
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
        # 留下小數點後兩位就好
        percent = round((b / a) * 100, 3)
        print("Before: ", a, "After: ", b, "Percent: ", percent, "%")

class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        # self.gpt.eval()
        return self

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

def combine_loss(logits: torch.Tensor, tokens: torch.Tensor, funnyscore: torch.Tensor, output_fc: torch.Tensor):
    output_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
    # humor // 1 ==1 > 1 , else == 0> 0
    funnyscore = funnyscore // 1
    fc_loss = nnf.binary_cross_entropy(output_fc.flatten(), funnyscore.flatten())
    reward = funnyscore.mean()
    loss = output_loss + fc_loss * 10 #- reward * 50
    return output_loss, fc_loss, loss

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None, entry_length=74, temperature=1.0,
                  stop_token: str = ".", ):
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
                # generated = model.gemma.base_model.model.model.embed_tokens(tokens)
                generated = model.falcon.model.embed_tokens(tokens)
        for i in range(entry_length):
            # outputs = model.gemma(inputs_embeds=generated)
            # outputs = model.gpt(inputs_embeds=generated)
            outputs = model.falcon(inputs_embeds=generated)
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
            # next_token_embed = model.gemma.base_model.model.model.embed_tokens(next_tokens.squeeze()).view( generated.shape[0], 1, -1)
            if model.LoRaActivated:
                next_token_embed = model.falcon.base_model.model.model.embed_tokens(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            else:
                next_token_embed = model.falcon.model.embed_tokens(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
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

def generate2(model, tokenizer, tokens=None, prompt=None, embed=None, entry_count=1, entry_length=74, top_p=0.8,
              temperature=1.0, stop_token: str = ".", ):
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
                # generated = model.gemma.base_model.model.model.embed_tokens(tokens)
                if model.LoRaActivated:
                    generated = model.falcon.base_model.model.model.embed_tokens(tokens)
                else:
                    generated = model.falcon.model.embed_tokens(tokens)
            for i in range(entry_length):
                # outputs = model.gemma(inputs_embeds=generated)
                # outputs = model.gpt(inputs_embeds=generated)
                outputs = model.falcon(inputs_embeds=generated)
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
                # next_token_embed = model.gemma.base_model.model.model.embed_tokens(next_token)
                # next_token_embed = model.gpt.transformer.wte(next_token)
                if model.LoRaActivated:
                    next_token_embed = model.falcon.base_model.model.model.embed_tokens(next_token)
                else:
                    next_token_embed = model.falcon.model.embed_tokens(next_token)
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
        if self.dataFrom == 'Oxford':
            prefix = torch.load('../../Oxford_HIC/ImageData/' + self.image_ids[item] + '.pt', weights_only=False)
        else:
            prefix = torch.load('../../Instagram/ImageData/' + self.dataFrom + '/' + self.image_ids[item] + '.pt',
                                weights_only=False)
        emotion, sentiment, humor = self.pad_emotion(item)
        # emotion, sentiment, humor = self.emotion[self.image_ids[item]], self.sentiment[self.image_ids[item]], self.humor[self.image_ids[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        funnyscore = self.funny_scores[item]
        return tokens, mask, prefix, item, emotion, sentiment, humor, funnyscore

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
        if dataFrom == 'Oxford':
            # self.emotions_data = pd.read_csv('../Data/Oxford_HIC/Minigpt4_Oxford.csv')
            self.emotions_data = pd.read_csv('../Data/Oxford_HIC/Minigpt4_Oxford_filter.csv')
            # self.emotion_data = pd.read_csv('../Data/Oxford_HIC/Minigpt4_Oxford_emotion_filter.csv')
            # self.sentiment_data = pd.read_csv('../Data/Oxford_HIC/Minigpt4_Oxford_sentiment_filter.csv')
            # self.humor_data = pd.read_csv('../Data/Oxford_HIC/Minigpt4_Oxford_humor_filter.csv')
        else:
            # self.emotions_data = pd.read_csv(f'../Data/Instagram/Minigpt4_{dataFrom}.csv')
            self.emotions_data = pd.read_csv(f'../Data/Instagram/Minigpt4_{dataFrom}_filter.csv')
            # self.emotion_data = pd.read_csv(f'../Data/Instagram/Minigpt4_{dataFrom}_emotion_filter.csv')
            # self.sentiment_data = pd.read_csv(f'../Data/Instagram/Minigpt4_{dataFrom}_sentiment_filter.csv')
            # self.humor_data = pd.read_csv(f'../Data/Instagram/Minigpt4_{dataFrom}_humor_filter.csv')
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
            self.emotion[image_id] = torch.tensor(self.tokenizer.encode(emotion, max_length=64, truncation=True),
                                                  dtype=torch.int64)
            self.sentiment[image_id] = torch.tensor(self.tokenizer.encode(sentiment, max_length=64, truncation=True),
                                                    dtype=torch.int64)
            self.humor[image_id] = torch.tensor(
                self.tokenizer.encode(humor, max_length=64, truncation=True, padding=True), dtype=torch.int64)

        # padding_emotion = 384 - (self.emotion_data.shape[1] - 1)
        # padding_sentiment = 384 - (self.sentiment_data.shape[1] - 1)
        # padding_humor = 768 - (self.humor_data.shape[1] - 1)
        # for i in range(self.emotion_data.shape[0]):
        #     image_id = self.emotion_data.iloc[i]['image_id']
        #     emotion = torch.tensor(self.emotion_data.iloc[i][1:].tolist())
        #     self.emotion[image_id] = torch.cat((emotion, torch.zeros(padding_emotion, dtype=torch.int64)))
        #     sentiment = torch.tensor(self.sentiment_data.iloc[i][1:].tolist())
        #     self.sentiment[image_id] = torch.cat((sentiment, torch.zeros(padding_sentiment, dtype=torch.int64)))
        #     humor = torch.tensor(self.humor_data.iloc[i][1:].tolist())
        #     self.humor[image_id] = torch.cat((humor, torch.zeros(padding_humor, dtype=torch.int64)))
        print(f"Train Data size: {len(self.captions_tokens)}")
        # self.filter_data_by_bleu(model, batch_size, bleu_threshold)

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
        return tokens, mask, prefix

    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        # self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        # self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-1B-Base")
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
trainData = '../Data/Oxford_HIC/parse/oxford_800_8_300_2_ViT-B_32_train.pkl'
testData = '../Data/Oxford_HIC/parse/oxford_800_8_300_2_ViT-B_32_test.pkl'
# trainData = '../Data/Instagram/parse/100up_only100_passlength_o_10_sonicdrivein_ViT-B_32_train.pkl'
# testData = '../Data/Instagram/parse/100up_only100_passlength_x_10_sonicdrivein_ViT-B_32_test.pkl'
prefix_length = 64
normalize_prefix = False
train_dataform = "Oxford"
test_dataform = "Oxford"
trainDataset = OxfordDataset(trainData, prefix_length, normalize_prefix=normalize_prefix, dataFrom = train_dataform)
testDataset = OxfordDataset(testData, prefix_length, normalize_prefix=normalize_prefix, dataFrom = test_dataform)

if train_dataform:
    ##################### oxford_300k #####################
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
    ##################### oxford_100k #####################
    # train_image = ['imgflip_34', 'bokete_3820', 'imgflip_0','imgflip_8', 'imgflip_15', 'imgflip_19', 'bokete_104530','imgflip_730', 'imgflip_130', 'imgflip_677']
    # train_text = ['You finish doing something at your friends house and look at your phone; 7 missed calls from your mom; 7 missed calls from your mom'
    #               ,'I\'m in my 50s!'
    #               ,'School; Memes'
    #               ,'image tagged in memes,one does not simply'
    #               ,'THAT; IS WHAT A GOOD MEME LOOKS LIKE'
    #               ,'NOT SURE IF PEOPLE ARE UPVOTING MEMES; OR USER NAMES'
    #               ,'It\'s a family night runaway.'
    #               ,'Chuck Norris doesn\'t go washroom; He goes washBOOM!'
    #               ,'SO YOUR TELLIN\' ME THAT SCHOOLS GOOD FOR YOU'
    #               ,'Y\'ALL GOT ANY MORE OF THEM; JOBS?']
    ##################### oxford_Top10_300k #####################
    # train_image = ['bokete_100136', 'bokete_100144', 'bokete_100174','bokete_100193', 'bokete_100268', 'bokete_100287', 'bokete_100295','bokete_10031', 'bokete_100498', 'bokete_24339']
    # train_text = ['The wax isn\'t dry yet.'
    #               ,'I\'m sorry to hear that.'
    #               ,'You didn\'t put that microphone in the register, did you?'
    #               ,'I\'m showing my brother the privilege of being the youngest.'
    #               ,'Are you ready to join us?'
    #               ,'He\'s cute, he\'s 100% capable of killing.'
    #               ,'The ground suddenly fell to the left.'
    #               ,'"Mama, there\'s something in the front mat!"'
    #               ,'"You don\'t have a dad?" "You don\'t have a mom?"'
    #               ,'This month, I\'ve only got this much to offer.']
    ##################### oxford_Top10_300k_mess #####################
    # train_image = ['bokete_100345', 'bokete_100360', 'bokete_100364','bokete_100193', 'bokete_100372', 'bokete_100432', 'bokete_100295','bokete_100453', 'bokete_100498', 'bokete_100459']
    # train_text = ['I had a dream about going to school, so I want to take a day off from school.'
    #               ,'I went to the woods for a jog, and there was a lot of spider webs.'
    #               ,'I\'d like to ask you a different color.'
    #               ,'I\'m showing my brother the privilege of being the youngest.'
    #               ,'Ah! You bumped into me in the morning!'
    #               ,'Are you sure it\'s your dad who left you when you were three?'
    #               ,'The ground suddenly fell to the left.'
    #               ,'In the first place, there\'s a problem with Snow White, who eats apples given to an old lady who looks so bad.'
    #               ,'It was at this time that they switched.'
    #               ,'Did you think it was corn?']
    ##################### oxford_Only1200_300k #####################
    # train_image = ['imgflip_0', 'imgflip_101', 'imgflip_1033','imgflip_11', 'imgflip_117', 'imgflip_16', 'imgflip_189','imgflip_23', 'imgflip_47', 'imgflip_504']
    # train_text = ['12 dollars; 11 dollars with 1 dollar shipping'
    #               ,'I HAD A GIRLFRIEND; AAAAAAND ITS GONE'
    #               ,'I POUR THE CEREAL AFTER I POUR THE MILK'
    #               ,'WAITING FOR MY PHONE TO GET  TO 100%'
    #               ,'You when you have over one test at school in a day'
    #               ,'IF SOMEONE WANTS TO KILL YOU; GO TO A LIVING ROOM'
    #               ,'you; eating 5 pounds of cheese; every day; your stomach'
    #               ,'Me:stands up to stretch my legs; The person who had been pushing my wheelchair for the last 26 years'
    #               ,'Me: Opens door for some fresh air; Everyone else in the submarine:'
    #               ,'THEY TOOK AWAY MY HAPPY MEAL I TOOK AWAY THEIR HAPPINESS']
    ##################### oxford_Only100_300k #####################
    # train_image = ['2spbgym', 'all-the-things', 'imgflip_0', 'imgflip_1033','imgflip_11', 'imgflip_117', 'imgflip_16', 'imgflip_189','imgflip_23', 'imgflip_504']
    # train_text = ['climb a mountain? pff, i have wings...'
    #               ,'go to a pizza buffet eat all the pizza'
    #               ,'12 dollars; 11 dollars with 1 dollar shipping'
    #               ,'I POUR MILK BEFORE CEREAL'
    #               ,'ME WAITING FOR MY INTERNET TO RECONNECT'
    #               ,'calling the teacher mom'
    #               ,'IF SOMEONE DIES IN THE LIVING ROOM... IS IT STILL CALLED THE LIVING ROOM?'
    #               ,'you; losing a few seconds of your life looking at this'
    #               ,'me: gets up and starts clapping because the chiefs won; the guy who has been pushing my wheelchair for 10 years'
    #               ,'THEY TOOK AWAY MY HAPPY MEAL I TOOK AWAY THEIR HAPPINESS']
    ##################### oxford_only10 #####################
    # train_image = ['2spbgym', 'all-the-things', 'imgflip_0', 'imgflip_1033','imgflip_11', 'imgflip_117', 'imgflip_16', 'imgflip_189','imgflip_23', 'imgflip_504']
    # train_text = ['i\'ll take it sure ur not chicken?',
    #               'avoid all the work',
    #               'due tomorrow; do tomorrow',
    #               'i want to speak to your manager',
    #               'school: *closes*; the kid who was in the bathroom:',
    #               'almost getting pass an extremely hard level but failing last minute',
    #               'you can\'t lose your mind; if you don\'t have one',
    #               'you; doing nothing; today; teacher',
    #               'students: *acting crazy*; teacher: pay attention! that kid named attention:',
    #               'they called me four eyes. i call them no eyes.']
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
            if len(train_image) == 10:
                break
    #######################################################
    tokens_list = []
    mask_list = []
    prefix_list = []
    train_emotion_list = []
    train_sentiment_list = []
    train_humor_list = []
    train_funnyscore = []
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
                tokens, mask, prefix, item, emotion, sentiment, humor, funnyscore = trainDataset[i]
                tokens_list.append(tokens)
                mask_list.append(mask)
                prefix_list.append(prefix)
                emotion, sentiment, humor = trainDataset.pad_emotion(item)
                train_emotion_list.append(emotion)
                train_sentiment_list.append(sentiment)
                train_humor_list.append(humor)
                train_funnyscore.append(funnyscore)
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
                    tokens, mask, prefix, item, emotion, sentiment, humor, funnyscore = trainDataset[i]
                    tokens_list.append(tokens)
                    mask_list.append(mask)
                    prefix_list.append(prefix)
                    train_gt.append(caption)
                    emotion, sentiment, humor = trainDataset.pad_emotion(item)
                    train_emotion_list.append(emotion)
                    train_sentiment_list.append(sentiment)
                    train_humor_list.append(humor)
                    train_funnyscore.append(funnyscore)
                    train_image_id_list.append(image_id)
    train_tokens = torch.stack(tokens_list).to(device)
    train_mask = torch.stack(mask_list).to(device)
    train_prefix = torch.stack(prefix_list).to(device)
    train_emotion = torch.stack(train_emotion_list).to(device)
    train_sentiment = torch.stack(train_sentiment_list).to(device)
    train_humor = torch.stack(train_humor_list).to(device)
    train_funnyscore = torch.tensor(train_funnyscore).to(device)
    print(train_tokens.shape, train_mask.shape, train_prefix.shape, len(train_image_id_list))
    print(train_emotion.shape, train_sentiment.shape, train_humor.shape, train_funnyscore.shape)

if test_dataform:
    ##################### oxford_300k #####################
    # test_image = ['imgflip_7', 'imgflip_32']
    # test_text = ['CHEESE; ME AT 3 AM; CHEESE; MY MOM WHO WAS WAITING; ME'
    #               ,'IS THIS A PIGEON?']
    ##################### oxford_100k #####################
    # test_image = ['imgflip_7', 'imgflip_32']
    # test_text = ['THE DOG FOOD; MY DOG; DOG FOOD; ME LOOKING AT HIM; MY DOG'
    #               ,'MATH; ME; IS THIS THE REASON I DROPPED OUT OF COLLEGE?']
    ##################### oxford_10 k #####################
    # test_image = ['bokete_111723', 'imgflip_57']
    # test_text = ['I\'m going to take care of you. I\'m going to take care of you. I\'m going to take care of you.'
    #               ,'WHAT\'S DONE IN THE DARK WILL ALWAYS COME OUT IN THE LIGHT; BUT THATS NONE OF MY BUSINESS']
    ##################### oxford_Top10_300k ###############
    # test_image = ['are-you-serious-face', 'bokete_24326']
    # test_text = ['you have windows 98 seriously?'
    #               ,'The answer is 10.']
    ##################### oxford_Top10_300k_mess ##########
    # test_image = ['imgflip_156', 'bokete_24326']
    # test_text = ['Friend: I just had a dream in which I married my crush! My dreams:'
    #               ,'The answer is 10.']
    ##################### oxford_Only1200_300k ############
    # test_image = ['imgflip_130', 'imgflip_659']
    # test_text = ['0 VIEWS 5 DISLIKES'
    #               ,'when the mobile game ad is so laggy that it crashes your game and you lose out on a reward:']
    ##################### oxford_Only100_300k #############
    # test_image = ['i-love-coloring-kid', 'imgflip_130']
    # test_text = ['she started writing notes !!'
    #               ,'WHEN YOUR FRIEND; DOSENT LIKE ROOT BEER']
    ##################### oxford_only10 ###################
    # test_image = ['bokete_100174', 'imgflip_834']
    # test_text = ['get out of my way. i\'ll do it.'
    #                 ,'me fully prepared for the test; question 1']
    ####################### default #######################
    test_image = []
    test_text = []
    test_emotion_list = []
    test_sentiment_list = []
    test_humor_list = []
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
    test_emotion_list = []
    test_sentiment_list = []
    test_humor_list = []
    test_funnyscore_list = []
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
                tokens, mask, prefix, item, emotion, sentiment, humor, funnyscore = testDataset[i]
                tokens_list.append(tokens)
                mask_list.append(mask)
                prefix_list.append(prefix)
                emotion, sentiment, humor = testDataset.pad_emotion(item)
                test_emotion_list.append(emotion)
                test_sentiment_list.append(sentiment)
                test_humor_list.append(humor)
                test_funnyscore_list.append(funnyscore)
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
                    tokens, mask, prefix, item, emotion, sentiment, humor, funnyscore = testDataset[i]
                    tokens_list.append(tokens)
                    mask_list.append(mask)
                    prefix_list.append(prefix)
                    emotion, sentiment, humor = testDataset.pad_emotion(item)
                    test_emotion_list.append(emotion)
                    test_sentiment_list.append(sentiment)
                    test_humor_list.append(humor)
                    test_funnyscore_list.append(funnyscore)
                    test_gt.append(caption)
                    test_image_id_list.append(image_id)

    test_tokens = torch.stack(tokens_list).to(device)
    test_mask = torch.stack(mask_list).to(device)
    test_prefix = torch.stack(prefix_list).to(device)
    test_emotion = torch.stack(test_emotion_list).to(device)
    test_sentiment = torch.stack(test_sentiment_list).to(device)
    test_humor = torch.stack(test_humor_list).to(device)
    test_funnyscore = torch.tensor(test_funnyscore_list).to(device)
    print(test_tokens.shape, test_mask.shape, test_prefix.shape)
    print(test_emotion.shape, test_sentiment.shape, test_humor.shape, test_funnyscore.shape)

model = ClipCaptionModel(mode='nn', prefix_length=prefix_length, clip_length=prefix_length, prefix_size=512, num_layers=8)
adapter = True
save_file = '20250316_oxford_800_8_300_2_ESH_filter_cross_concat'
i = 13
if adapter :
    model.load_state_dict(torch.load(f'./Model/{save_file}/checkpoint-{i:03d}.pt'))
    model_adapt = ClipCaptionModel(mode='lora', prefix_length=prefix_length, clip_length=prefix_length, prefix_size=512, num_layers=8)

    def count_trainable_parameters(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


    def weightToLora(fullModel, newModel):
        for name, param in fullModel.named_parameters():
            newModel.state_dict()[name].copy_(param.data)
        for name, param in newModel.named_parameters():
            if 'falcon' not in name:
                if "lora_" not in name:  # 只讓 LoRA 參數訓練
                    if 'bias' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                    if 'funnyscore' in name:
                        param.requires_grad = True
                else:
                    param.requires_grad = True
        for name, param in newModel.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        return newModel

    a = count_trainable_parameters(model)
    model = weightToLora(model, model_adapt)
    del model_adapt
    gc.collect()
    b = count_trainable_parameters(model)
    percent = round((b / a) * 100, 3)
    print("Before: ", a, "After: ", b, "Percent: ", percent, "%")

    # model.activateLoRa()
    #
    # b = count_trainable_parameters(model)
    # percent = round((b / a) * 100, 3)
    # print("Before: ", a, "After: ", b, "Percent: ", percent, "%")

save_file = '20250322_100up_only100_passlength_10_base_0316_oxford_800_8_300_2_ESH_filter_cross_concat_combineLoss_falconLoRa'
for i in range(20):
    if os.path.exists(f'./Model/{save_file}/checkpoint-{i + 1:03d}.pt'):
        model.load_state_dict(torch.load(f'./Model/{save_file}/checkpoint-{i + 1:03d}.pt'))
        model = model.eval()
        model = model.to(device, dtype=torch.bfloat16)
        pred = Predictor(prefix_length, cp_num=i + 1, train_caption=train_caption, test_caption=test_caption,
                         train_image_id_list=train_image_id_list, test_image_id_list=test_image_id_list)
        print(f"Model {i + 1:03d} loaded.")
        pred.predict("test", test_tokens, test_mask, test_prefix, test_gt, model, test_emotion, test_sentiment, test_humor, test_funnyscore)
        pred.predict("train", train_tokens, train_mask, train_prefix, train_gt, model, train_emotion, train_sentiment, train_humor, train_funnyscore)

AllCaption = pd.DataFrame()
for i in range(20):
    if os.path.exists(f'./Model/{save_file}/checkpoint-{i + 1:03d}.pt'):
        # df = pd.read_csv(f'./Model/{save_file}/{save_file}_test_{i + 1:03d}.csv')
        df = pd.read_csv(f'./Model/{save_file}/test_{i + 1:03d}.csv')
        # if adapter:
        #     textAndLoss = df[['text', 'loss', 'caption_loss', 'fc_loss', 'fitCount', 'gtNum', 'fluency', 'diversity', 'bleu1', 'bleu2', 'bleu3', 'bleu4']]
        # else:
        textAndLoss = df[['text', 'loss', 'fitCount', 'gtNum', 'fluency', 'diversity', 'bleu1', 'bleu2', 'bleu3', 'bleu4']]
        if AllCaption.empty:
            AllCaption = df[['Name', 'image_id']]
        AllCaption = pd.concat([AllCaption, textAndLoss], axis=1)
# AllCaption.to_csv(f'./Model/{save_file}/{save_file}_test_all.csv', float_format='%.15f', index=False)
AllCaption.to_csv(f'./Model/{save_file}/test_all.csv', float_format='%.15f', index=False)

