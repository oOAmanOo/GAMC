import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Gemma2ForCausalLM, \
    TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

from extractor import addImagePath, textExtraction, imageExtraction, textExtractReverse, textExtractReverse_embedd
eps = torch.finfo(torch.bfloat16).eps

### 官方的Gemma #########################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2b = 2304, 9b = 3584, 27b = 4608
gemma_hiddenstate_size = 2304
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
gemmaConfig = AutoConfig.from_pretrained('google/gemma-2-2b-it')
### gemma float32 / bfloat16
gemma = Gemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16)
# gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16)

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
a = count_trainable_parameters(gemma)
gemma = get_peft_model(gemma, LORAconfig)
b = count_trainable_parameters(gemma)
#留下小數點後兩位就好
percent = round((b / a) * 100, 3)
print("Before: ", a, "After: ", b, "Percent: ", percent, "%")
del a, b, percent
gc.collect()
########################################################################################################
class self_image(nn.Module):
    def __init__(self):
        super(self_image, self).__init__()
        # self attention
        self.selfAttentionMultihead = nn.MultiheadAttention(768, 1)
        self.selfAttentionLayerNorm = nn.LayerNorm(768, eps=eps)
        self.selfAttentionLinear1 = nn.Linear(768, 768)
        self.selfAttentionRelu = nn.ReLU()
        self.selfAttentionLinear2 = nn.Linear(768, 768)
        self.selfAttentionLayerNorm2 = nn.LayerNorm(768, eps=eps)

        self.prefix_const = nn.Parameter(torch.randn(64, 768), requires_grad=True)
        # co-attention text
        self.coAttentionTextMultihead = nn.MultiheadAttention(768, 8)
        self.coAttentionTextLinear1 = nn.Linear(768, 768)
        self.coAttentionTextRelu = nn.ReLU()
        self.coAttentionTextLinear2 = nn.Linear(768, 768)
        self.coAttentionTextLayerNorm = nn.LayerNorm(768, eps=eps)

        # co-attention image
        self.coAttentionImageMultihead = nn.MultiheadAttention(768, 8)
        self.coAttentionImageLinear1 = nn.Linear(768, 768)
        self.coAttentionImageRelu = nn.ReLU()
        self.coAttentionImageLinear2 = nn.Linear(768, 768)
        self.coAttentionImageLayerNorm = nn.LayerNorm(768, eps=eps)

        # feed forward
        self.feedForwardLinear1 = nn.Linear(768, 768)
        self.feedForwardRelu = nn.ReLU()
        self.feedForwardLinear2 = nn.Linear(768, 768)

    def forward(self, image):
        # self attention module
        self_temp = self.selfAttentionMultihead(image, image, image)[0]
        self_temp = self.selfAttentionLayerNorm(self_temp + image)
        self_out = self.selfAttentionLinear1(self_temp)
        self_out = self.selfAttentionRelu(self_out)
        self_out = self.selfAttentionLinear2(self_out)
        self_out = self.selfAttentionLayerNorm2(self_out + self_temp)

        prefix = self.prefix_const.unsqueeze(0).expand(image.shape[1], -1, -1).transpose(0, 1).to(device).to(torch.bfloat16)
        # co-attention image module
        visual_attending_textual = self.coAttentionTextMultihead(self_out, prefix, prefix)[0]
        visual_attending_textual = self.coAttentionTextLinear1(visual_attending_textual)
        visual_attending_textual = self.coAttentionTextRelu(visual_attending_textual)
        visual_attending_textual = self.coAttentionTextLinear2(visual_attending_textual)
        visual_attending_textual = self.coAttentionTextLayerNorm(visual_attending_textual + self_out)

        # co-attention text module
        textual_attending_visual = self.coAttentionTextMultihead(prefix, self_out, self_out)[0]
        textual_attending_visual = self.coAttentionTextLinear1(textual_attending_visual)
        textual_attending_visual = self.coAttentionTextRelu(textual_attending_visual)
        textual_attending_visual = self.coAttentionTextLinear2(textual_attending_visual)
        textual_attending_visual = self.coAttentionTextLayerNorm(textual_attending_visual + prefix)

        output = self.feedForwardLinear1(visual_attending_textual + textual_attending_visual)
        output = self.feedForwardRelu(output)
        output = self.feedForwardLinear2(output)

        return output

class multi_text(nn.Module):
    def __init__(self):
        super(multi_text, self).__init__()
        # multihead attention
        self.multiheadAttentionMultihead = nn.MultiheadAttention(768, 8)
        self.multiheadAttentionLinear1 = nn.Linear(768, 768)
        self.multiheadAttentionRelu = nn.ReLU()
        self.multiheadAttentionLinear2 = nn.Linear(768, 768)
        self.multiheadAttentionLayerNorm = nn.LayerNorm(768, eps=eps)

    def forward(self, text):

        # multihead attention module
        multi_out = self.multiheadAttentionMultihead(text, text, text)[0]
        multi_out = self.multiheadAttentionLinear1(multi_out)
        multi_out = self.multiheadAttentionRelu(multi_out)
        multi_out = self.multiheadAttentionLinear2(multi_out)
        multi_out = self.multiheadAttentionLayerNorm(multi_out + text)

        return multi_out


class Former(nn.Module):
    def __init__(self, depth=12):
        super(Former, self).__init__()
        self.layers_self = nn.ModuleList([self_image() for _ in range(depth)])
        self.layers_multi = nn.ModuleList([multi_text() for _ in range(depth)])
        # self.layers_co_attention = nn.ModuleList([co_attention() for _ in range(depth)])
    def image (self, image):
        for self_layer in self.layers_self:
            image = self_layer(image)
        return image
    def text (self, text):
        for multi_layer in self.layers_multi:
            text = multi_layer(text)
        return text
    def forward(self, text, image):

        text = text.transpose(0, 1)
        text = self.text(text)
        image = image.transpose(0, 1)
        image = self.image(image)
        return image, text

class Generator(nn.Module):
    def __init__(self, depth=12):
        super(Generator, self).__init__()
        # gemma
        self.gemmaLinearMaxTokens = nn.Linear(64, 32)
        # self.gemmaLinearBefore = nn.Linear(768, gemmaConfig.vocab_size)
        self.gemmaSoftmax = nn.Softmax(dim=2)
        self.gemmalinearAfter = nn.Linear(gemmaConfig.vocab_size, 768)
        # embedding
        self.gemmaLinearBefore = nn.Linear(768, gemma_hiddenstate_size)
        # feed forward
        self.feedForwardLinear = nn.Linear(768, 768)
        self.feedForwardLayerNorm = nn.LayerNorm(768, eps=eps)
        # funny score
        self.FunnyScorelinear1 = nn.Linear(768, 1)
        self.FunnyScorelinear2 = nn.Linear(64, 1)

    def forward(self, Former, image, text_id):
        ##############################################   generate ##############################################
        image = image.transpose(0, 1)
        image_G = Former.image(image)

        image_G = image_G.transpose(0, 1)
        # image_GG = self.gemmaLinearMaxTokens(image_G.transpose(1, 2)).transpose(1, 2)
        # image_GG = self.gemmaLinearBefore(image_GG)
        # image_GG = self.gemmaSoftmax(image_GG + eps)
        # get max value of each row, total 32*64
        image_GG = self.gemmaLinearBefore(image_G)
        # with torch.no_grad():
        # top_k_values, top_k_indices = torch.topk(image_GG, 1, dim=2, largest=True)
        # loss, logits = textExtractReverse(gemma, tokenizer, top_k_indices, text_id)
        loss, logits = textExtractReverse_embedd(gemma, image_GG, text_id)
        # ###########################################   funny score   ###########################################
        # text = self.gemmalinearAfter(logits.to(torch.bfloat16))
        # text = text.transpose(0, 1)
        # text_F = Former.text(text).transpose(0, 1)
        # ######################### funny score #########################
        # feature_fusion = image_G + text_F  # visual_attending_textual + textual_attending_visual
        # feature_fusionFF = self.feedForwardLinear(feature_fusion)
        # feature_fusion_final = self.feedForwardLayerNorm(feature_fusion + feature_fusionFF)
        # feature_fusion_final = feature_fusion_final.squeeze(-1)
        # feature_fusion_final = feature_fusion_final
        # output_funny_score = self.FunnyScorelinear1(feature_fusion_final).squeeze(-1)
        # output_funny_score = self.FunnyScorelinear2(output_funny_score).squeeze(-1)
        # #######################################################################################################
        output_funny_score = 0
        return loss, output_funny_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############# load  model #############
NetFormer = Former().to(torch.bfloat16).to(device)
Generator = Generator().to(torch.bfloat16).to(device)
#######################################
checkpoint_folder = '20241205_LoRa_embedd_noPrompt_onlyGemmaLoss_base_20241201_coAttention_temp'
checkpoint_Former = 'D:MemeGAN/Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetFormer_5.pth'
checkpoint_Generator = 'D:MemeGAN/Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetLLM_5.pth'
checkpoint_Gemma = 'D:MemeGAN/Model/' + checkpoint_folder + '/' + checkpoint_folder + '_Gemma_5.pth'
#######################################
checkpoint_Former = torch.load(checkpoint_Former)
checkpoint_Generator = torch.load(checkpoint_Generator)
NetFormer.load_state_dict(checkpoint_Former['model_state_dict'])
Generator.load_state_dict(checkpoint_Generator['model_state_dict'])
gemma.load_state_dict(torch.load(checkpoint_Gemma)['model_state_dict'])
#######################################

# generate
NetFormer.eval()
Generator.eval()
gemma.eval()

# image = imageExtraction("./test_img.jpg").to(torch.bfloat16)
# test_gt = ['you all loved it so i brought it back']
# image = imageExtraction("./test_img2.jpg").to(torch.bfloat16)
# test_gt = ['Can’t believe this is real (still) life. The Saucy Nugg legacy is forever cemented in oil on canvas (2024). Thank you']
image = torch.load('../../Oxford_HIC/ImageData/bokete_9678.pt', weights_only=False).unsqueeze(0).to(torch.bfloat16)
test_gt = ['I\'m like, \"What is this big?\"']

text_id = textExtraction(tokenizer, gemmaConfig, test_gt)
gemma_loss, output_funny_score = Generator(NetFormer, image.to(device), text_id.to(device))
print(gemma_loss,output_funny_score)
