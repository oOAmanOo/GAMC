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
batch_size = 15
dirPath = '../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
data = pd.read_csv(dirPath)
print("shape of data: ", data.shape)
data = data.sample(n=2000, random_state=42, replace=True).reset_index(drop=True)
print("sample of data: ", data.shape)
train, test = train_test_split(data, test_size=0.2, random_state=42)
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

prompt_gemma = "Create memetic post on Instagram."
# prompt_gemma = "Answer with only 100 words, only answer, no explain. Write a humor memetic post for Instagram with the following elements: "
prompt_gemma = tokenizer(prompt_gemma, padding_side="right", truncation=True, padding='max_length', max_length=64, return_tensors='pt').to(device)
text_embedding = nn.Embedding(gemmaConfig.vocab_size, 768).to(device)
prompt_gemma = text_embedding(prompt_gemma['input_ids']).to(device)
prompt_gemma = prompt_gemma.squeeze(1).expand(batch_size, -1, -1).to(torch.bfloat16)
# print(prompt_gemma.shape)
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

        # prefix = self.prefix_const.unsqueeze(0).expand(image.shape[1], -1, -1).transpose(0, 1).to(device).to(torch.bfloat16)
        # # co-attention image module
        # visual_attending_textual = self.coAttentionTextMultihead(self_out, prefix, prefix)[0]
        # visual_attending_textual = self.coAttentionTextLinear1(visual_attending_textual)
        # visual_attending_textual = self.coAttentionTextRelu(visual_attending_textual)
        # visual_attending_textual = self.coAttentionTextLinear2(visual_attending_textual)
        # visual_attending_textual = self.coAttentionTextLayerNorm(visual_attending_textual + self_out)
        #
        # # co-attention text module
        # textual_attending_visual = self.coAttentionTextMultihead(prefix, self_out, self_out)[0]
        # textual_attending_visual = self.coAttentionTextLinear1(textual_attending_visual)
        # textual_attending_visual = self.coAttentionTextRelu(textual_attending_visual)
        # textual_attending_visual = self.coAttentionTextLinear2(textual_attending_visual)
        # textual_attending_visual = self.coAttentionTextLayerNorm(textual_attending_visual + prefix)
        #
        # output = self.feedForwardLinear1(visual_attending_textual + textual_attending_visual)
        # output = self.feedForwardRelu(output)
        # output = self.feedForwardLinear2(output)

        return self_out

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
    def __init__(self, Former, gemma, depth=12):
        super(Generator, self).__init__()
        self.Former = Former
        self.gemma = gemma
        # gemma
        self.gemmaLinearMaxTokens = nn.Linear(64, 32)
        self.gemmaLinearBefore = nn.Linear(768, gemmaConfig.vocab_size)
        self.gemmaSoftmax = nn.Softmax(dim=2)
        # embedding
        # self.gemmaLinearBefore = nn.Linear(768, gemma_hiddenstate_size)
        # self.gemmaLinearBefore = nn.Linear(1536, gemma_hiddenstate_size)
        # feed forward
        self.feedForwardLinear = nn.Linear(768, 768)
        self.feedForwardLayerNorm = nn.LayerNorm(768, eps=eps)
        # funny score
        self.FunnyScorelinear1 = nn.Linear(768, 1)
        self.FunnyScorelinear2 = nn.Linear(64, 1)

    def forward(self, image, text_id, prompt_gemma2):
        ##############################################   generate ##############################################
        image = image.transpose(0, 1)
        image_G = self.Former.image(image)

        image_G = image_G.transpose(0, 1)
        image_GG = self.gemmaLinearMaxTokens(image_G.transpose(1, 2)).transpose(1, 2)
        image_GG = self.gemmaLinearBefore(image_GG)
        image_GG = self.gemmaSoftmax(image_GG + eps)
        # get max value of each row, total 32*64

        # with torch.no_grad():
        top_k_values, top_k_indices = torch.topk(image_GG, 1, dim=2, largest=True)
        loss, logits = textExtractReverse(gemma, tokenizer, top_k_indices, text_id)
        # image_GG = torch.cat((prompt_gemma2, image_G), dim=-1) # 8 + 64 = 72
        # image_GG = self.gemmaLinearBefore(image_GG)
        # loss, logits = textExtractReverse_embedd(gemma, image_GG, text_id)
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
        return loss, logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############# load  model #############
NetFormer = Former().to(torch.bfloat16).to(device)
Generator = Generator(Former= NetFormer, gemma=gemma).to(torch.bfloat16).to(device)
#######################################
checkpoint_folder = '20241211_LoRa_id_shortPrompt_CE_base_20241201_wo_coAttention_temp'
checkpoint_Generator = './Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetLLM_1.pth'
#######################################
checkpoint_Generator = torch.load(checkpoint_Generator)
Generator.load_state_dict(checkpoint_Generator['model_state_dict'])
#######################################

def loss_function(logits, text_id):
    # loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
    # logits = logits[:, :-1, :].contiguous()
    # text_id = text_id[:, 1:].contiguous()
    logits = logits[:, 100:, :].contiguous()
    loss = nn.CrossEntropyLoss()(logits.view(-1, gemmaConfig.vocab_size), text_id.view(-1))
    return loss
def loss_function_next(logits, text_id):
    # loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
    logits = logits[:, 100:, :].contiguous()
    logits = logits[:, :-1, :].contiguous()
    text_id = text_id[:, 1:].contiguous()
    loss = nn.CrossEntropyLoss()(logits.view(-1, gemmaConfig.vocab_size), text_id.view(-1))
    return loss

# generate
Generator.eval()

image = imageExtraction("./test_img.jpg").expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['you all loved it so i brought it back']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
#
image = imageExtraction("./test_img2.jpg").expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['Can’t believe this is real (still) life. The Saucy Nugg legacy is forever cemented in oil on canvas (2024). Thank you']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())



print('=======================  sample1 =========================')
print(train[train['image_id'] == 'bokete_51227'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/bokete_51227.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['Matsui found a porn actress in the audience.']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
print('=======================  sample2 =========================')
print(train[train['image_id'] == 'bokete_3820'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/bokete_3820.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['I\'m in my 50s!']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
print('=======================  sample3 =========================')
print(train[train['image_id'] == 'imgflip_0'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/imgflip_0.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['Getting hurt from cutting open your leg; Getting hurt from taking off a bandaid']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
print('=======================  sample4 =========================')
print(train[train['image_id'] == 'imgflip_8'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/imgflip_8.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['image tagged in memes,one does not simply']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
print('=======================  sample5 =========================')
print(train[train['image_id'] == 'imgflip_15'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/imgflip_15.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['KIDS WHEN THEIR PARENTS GIVE THEM; "THE TALK"']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
print('=======================  sample6 =========================')
print(train[train['image_id'] == 'imgflip_19'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/imgflip_19.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['NOT SURE IF PEOPLE ARE UPVOTING MEMES; OR USER NAMES']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
print('=======================  sample7 =========================')
print(train[train['image_id'] == 'bokete_104530'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/bokete_104530.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['It\'s a family night runaway.']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
print('=======================  sample8 =========================')
print(train[train['image_id'] == 'imgflip_730'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/imgflip_730.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['CHUCK IS THE GOOD TYPE OF SCUMBAG; CUZ HE ONLY ROASTS YOU FROM YOUR INSIDES']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
print('=======================  sample9 =========================')
print(train[train['image_id'] == 'imgflip_130'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/imgflip_130.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['SO YOUR TELLIN\' ME THAT SCHOOLS GOOD FOR YOU']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
print('=======================  sample10 =========================')
print(train[train['image_id'] == 'imgflip_677'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageData/imgflip_677.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
test_gt = ['Y\'ALL GOT ANY MORE OF THEM; JOBS?']
text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
loss = loss_function(logits, text_id.to(device))
loss_next = loss_function_next(logits, text_id.to(device))
print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())

