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

class OxfordDataset(torch.utils.data.Dataset):
    def __init__(self, text, image, funny_score):
        self.text = text
        self.image = image
        self.funny_score = funny_score

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        imageData = torch.load('../../Oxford_HIC/ImageData/'+ self.image[idx] +'.pt', weights_only=False)
        # all dtype to torch.float16
        imageData = imageData
        return self.text[idx], imageData, self.funny_score[idx]


def train():
    epochs = 200
    batch_size = 15
    optimizer_Former_lr = 1e-5
    save_name = '20241208_LoRa_embedd_smallPrompt_concat_onlyGemmaLoss_base_20241201_coAttention_temp'
    if not os.path.exists('./Model/' + save_name):
        os.makedirs('./Model/' + save_name)
        os.makedirs('D:/MemeGAN/Model/' + save_name)
    ################ right ################
    # twoModel = False
    # checkpoint_folder = '20241201_wo_coAttention_temp'
    # checkpoint_Former = 'D:/MemeGAN/Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetFormer_83.pth'
    ################ left  ################
    twoModel = False
    checkpoint_folder = '20241201_coAttention_temp'
    checkpoint_Former = 'D:/MemeGAN/Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetFormer_140.pth'
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
    data = data.sample(n=2000, random_state=42, replace=True).reset_index(drop=True)
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

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
    prompt_gemma = tokenizer(prompt_gemma, padding_side="right", truncation=True, padding='max_length', max_length=64, return_tensors='pt').to(device)
    text_embedding = nn.Embedding(gemmaConfig.vocab_size, 768).to(device)
    prompt_gemma = text_embedding(prompt_gemma['input_ids']).to(device)
    prompt_gemma = prompt_gemma.squeeze(1).expand(batch_size, -1, -1).to(torch.bfloat16)
    # print(prompt_gemma.shape)
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
            # self.gemmaLinearBefore = nn.Linear(768, gemma_hiddenstate_size)
            self.gemmaLinearBefore = nn.Linear(1536, gemma_hiddenstate_size)
            # feed forward
            self.feedForwardLinear = nn.Linear(768, 768)
            self.feedForwardLayerNorm = nn.LayerNorm(768, eps=eps)
            # funny score
            self.FunnyScorelinear1 = nn.Linear(768, 1)
            self.FunnyScorelinear2 = nn.Linear(64, 1)

        def forward(self, Former, image, text_id, prompt_gemma2):
            ##############################################   generate ##############################################
            image = image.transpose(0, 1)
            image_G = Former.image(image)

            image_G = image_G.transpose(0, 1)
            # image_GG = self.gemmaLinearMaxTokens(image_G.transpose(1, 2)).transpose(1, 2)
            # image_GG = self.gemmaLinearBefore(image_GG)
            # image_GG = self.gemmaSoftmax(image_GG + eps)
            # get max value of each row, total 32*64

            # with torch.no_grad():
            # top_k_values, top_k_indices = torch.topk(image_GG, 1, dim=2, largest=True)
            # loss, logits = textExtractReverse(gemma, tokenizer, top_k_indices, text_id)
            image_GG = torch.cat((prompt_gemma2, image_G), dim=-1) # 8 + 64 = 72
            image_GG = self.gemmaLinearBefore(image_GG)
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

    NetFormer = Former().to(torch.bfloat16).to(device)
    Generator = Generator().to(torch.bfloat16).to(device)
    optimizer_Former = optim.Adam(Generator.parameters(), lr=optimizer_Former_lr)

    train_losses_Former = []
    test_losses_Former = []
    save = []
    present_epoch = 1
    best_train_loss_Former = 9999999999
    best_test_loss_Former = 9999999999
    gemma_loss_list = []
    fc_loss_list = []

    checkpoint_Former = torch.load(checkpoint_Former)
    NetFormer.load_state_dict(checkpoint_Former['model_state_dict'])

    if twoModel:
        checkpoint_Generator = torch.load(0)
        Generator.load_state_dict(checkpoint_Generator['model_state_dict'])
        present_epoch = checkpoint_Former['epoch'] + 1
        del checkpoint_Generator

    del checkpoint_Former
    gc.collect()

    # def loss_function(output, labels, funny_score, target_funny_score):
    #     # match loss
    #     if output.shape[1] > labels.shape[1]:
    #         padding = torch.zeros(output.shape[0], output.shape[1] - labels.shape[1]).to(device)
    #         labels = torch.cat((labels, padding), dim=1)
    #     elif output.shape[1] < labels.shape[1]:
    #         padding = torch.zeros(output.shape[0], labels.shape[1] - output.shape[1]).to(device)
    #         output = torch.cat((output, padding), dim=1)
    #     gemmaLoss = (output != labels).float().mean()
    #     fc_loss = nn.MSELoss()(funny_score, target_funny_score)
    #     gemma_loss_list.append(gemmaLoss.item())
    #     fc_loss_list.append(fc_loss.item())
    #     loss = gemmaLoss + fc_loss
    #
    #     return loss

    def loss_function(gemmaLoss, funny_score, target_funny_score):
        # if gemma_logits.shape[1] > text_id.shape[1]:
        #     gemma_logits = gemma_logits[:, :text_id.shape[1], :]
        # elif gemma_logits.shape[1] < text_id.shape[1]:
        #     padding = torch.zeros(gemma_logits.shape[0], text_id.shape[1] - gemma_logits.shape[1], gemma_logits.shape[2])
        #     gemma_logits = torch.cat((gemma_logits, padding), dim=1)
        # gemmaLoss = nn.CrossEntropyLoss()(gemma_logits.view(-1, gemmaConfig.vocab_size), text_id.view(-1))

        # fc_loss = nn.MSELoss()(funny_score, target_funny_score)
        # gemma_loss_list.append(gemmaLoss.item())
        # fc_loss_list.append(fc_loss.item())
        # loss = gemmaLoss + fc_loss * 500
        loss = gemmaLoss
        return loss

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        print("---------------------------------------- epoch " + str(
            epoch + present_epoch) + " ---------------------------------------")
        train_loss_Former = 0
        test_loss_Former = 0
        ###################################### Train ######################################
        with tqdm(train_loader, unit="batch", leave=True) as tepoch:
            for idx, (text, image, funny_score) in enumerate(tepoch):
                text_id = textExtraction(tokenizer, gemmaConfig, text)
                image = image.to(torch.bfloat16)
                funny_score = funny_score.to(torch.bfloat16)
                optimizer_Former.zero_grad()
                gemma_loss, output_funny_score = Generator(NetFormer, image.to(device), text_id.to(device), prompt_gemma.detach())
                loss = loss_function(gemma_loss, output_funny_score, funny_score.to(device))
                # loss.requires_grad = True
                loss.backward()
                optimizer_Former.step()
                train_loss_Former += loss.item()
                tepoch.set_postfix(loss=train_loss_Former / (idx + 1))
                ##########################################################################
                # seperateLoss_data = pd.DataFrame()
                # seperateLoss_data['gemma_loss'] = gemma_loss_list
                # seperateLoss_data['fc_loss'] = fc_loss_list
                # seperateLoss_data.to_csv('./Model/' + save_name + '/' + save_name + '_seperateLoss.csv',
                #                          index=False)
                ##########################################################################
        train_loss_Former /= len(train_loader)
        train_losses_Former.append(train_loss_Former)
        ##################################### Test ######################################
        with tqdm(test_loader, unit="batch", leave=True) as tepoch:
            for idx, (text, image, funny_score) in enumerate(tepoch):
                text_id = textExtraction(tokenizer, gemmaConfig, text)
                image = image.to(torch.bfloat16)
                funny_score = funny_score.to(torch.bfloat16)
                gemma_loss, output_funny_score = Generator(NetFormer, image.to(device), text_id.to(device), prompt_gemma)
                loss = loss_function(gemma_loss, output_funny_score, funny_score.to(device))
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
                'model_state_dict': NetFormer.state_dict(),
                'optimizer_state_dict': optimizer_Former.state_dict(),
                'loss': loss,
            }, './Model/' + save_name + '/' + save_name + '_NetFormer_' + str(epoch + present_epoch) + '.pth')
            torch.save({
                'epoch': epoch + present_epoch,
                'model_state_dict': Generator.state_dict(),
                'optimizer_state_dict': optimizer_Former.state_dict(),
                'loss': loss,
            }, './Model/' + save_name + '/' + save_name + '_NetLLM_' + str(epoch + present_epoch) + '.pth')
            torch.save({
                'epoch': epoch + present_epoch,
                'model_state_dict': gemma.state_dict(),
                'optimizer_state_dict': optimizer_Former.state_dict(),
                'loss': loss,
            }, './Model/' + save_name + '/' + save_name + '_Gemma_' + str(epoch + present_epoch) + '.pth')
            save.append("V")
        else:
            save.append(" ")

        loss_data = pd.DataFrame()

        # loss_data['epoch'] = list(range(1, epoch + present_epoch + 1))
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