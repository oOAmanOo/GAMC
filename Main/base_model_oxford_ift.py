import gc
import os
import pandas as pd
from functorch.dim import use_c
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import BertLMHeadModel, BertTokenizer
# from local_gemma import LocalGemma2ForCausalLM

from extractor import addImagePath, textExtraction, imageExtraction, textExtractReverse, textExtraction_IFT
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
    checkpoint = False
    load_name = '20241110_gemma_generate_onlyd'
    load_num = 0
    epochs = 30
    batch_size = 40
    optimizer_F_lr = 1e-5
    save_name = 'ift_test'
    if not os.path.exists('./Model/' + save_name):
        os.makedirs('./Model/' + save_name)


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
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=20)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=20)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    ### 官方的Gemma #########################################################################################
    Fformer = BertLMHeadModel.from_pretrained("bert-base-uncased", is_decoder=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained('bert-base-uncased')
    # 2b = 2304, 9b = 3584, 27b = 4608
    # gemma_hiddenstate_size = 2304
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    # config = AutoConfig.from_pretrained('google/gemma-2-2b-it')
    # gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16)
    ########################################################################################################
    class Prefix(nn.Module):
        def __init__(self):
            super(Prefix, self).__init__()
            # init
            self.Fformer = Fformer
            self.Fformer.resize_token_embeddings(len(tokenizer))
            self.query_tokens = nn.Parameter(torch.zeros(1, 64, 768), requires_grad=True)
            self.query_tokens.data.normal_(mean=0.0, std=0.02)

            # co-attention const 10
            self.prefix_const = nn.Parameter(torch.randn(64, 768), requires_grad=True)
            self.coAttentionMultihead = nn.MultiheadAttention(768, 1)
            self.coAttentionLinear1 = nn.Linear(768, 768)
            self.coAttentionRelu = nn.ReLU()
            self.coAttentionLinear2 = nn.Linear(768, 768)
            self.coAttentionLayerNorm = nn.LayerNorm(768, eps=eps)

            # # co-attention const 10
            # self.prefix_const = nn.Parameter(torch.randn(64, 768), requires_grad=True)
            # self.coAttentionTextMultihead = nn.MultiheadAttention(768, 1)
            # self.coAttentionTextLinear1 = nn.Linear(768, 768)
            # self.coAttentionTextRelu = nn.ReLU()
            # self.coAttentionTextLinear2 = nn.Linear(768, 768)
            # self.coAttentionTextLayerNorm = nn.LayerNorm(768, eps=eps)
            #
            # # co-attention image
            # self.coAttentionImageMultihead = nn.MultiheadAttention(768, 1)
            # self.coAttentionImageLinear1 = nn.Linear(768, 768)
            # self.coAttentionImageRelu = nn.ReLU()
            # self.coAttentionImageLinear2 = nn.Linear(768, 768)
            # self.coAttentionImageLayerNorm = nn.LayerNorm(768, eps=eps)

            # feed forward
            self.feedForwardLinear = nn.Linear(768, 768)
            self.feedForwardLayerNorm = nn.LayerNorm(768, eps=eps)

        def forward(self, text, image):
            text_output = self.Fformer.bert(
                inputs_embeds=text,
                return_dict=True,
            )
            text_feat = nn.functional.normalize(text_output['last_hidden_state'], p=2, dim=-1)

            image_atts = torch.ones(image.shape[:1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image.shape[0], -1, -1).to(device).to(torch.bfloat16)

            query_output = self.Fformer.bert(
                inputs_embeds=query_tokens,
                use_cache=True,
                return_dict=True,
            )
            image_hidden = query_output['last_hidden_state'].transpose(0, 1)
            prefix = self.prefix_const.unsqueeze(0).expand(image.shape[0], -1, -1).transpose(0, 1).to(device).to(torch.bfloat16)

            # co-attention image module
            visual_attending_textual = self.coAttentionMultihead(image_hidden, prefix, prefix)[0]
            visual_attending_textual = self.coAttentionLinear1(visual_attending_textual)
            visual_attending_textual = self.coAttentionRelu(visual_attending_textual)
            visual_attending_textual = self.coAttentionLinear2(visual_attending_textual)
            visual_attending_textual = self.coAttentionLayerNorm(visual_attending_textual + image_hidden)

            # co-attention text module
            textual_attending_visual = self.coAttentionMultihead(prefix, image_hidden, image_hidden)[0]
            textual_attending_visual = self.coAttentionLinear1(textual_attending_visual)
            textual_attending_visual = self.coAttentionRelu(textual_attending_visual)
            textual_attending_visual = self.coAttentionLinear2(textual_attending_visual)
            textual_attending_visual = self.coAttentionLayerNorm(textual_attending_visual + prefix)

            # feature fusion
            feature_fusion = visual_attending_textual + textual_attending_visual
            feature_fusionFF = self.feedForwardLinear(feature_fusion)
            feature_fusion_final = self.feedForwardLayerNorm(feature_fusion + feature_fusionFF).transpose(0, 1)

            image_feat = nn.functional.normalize(feature_fusion_final, p=2, dim=-1)

            return text_feat, image_feat

    class IFormer(nn.Module):
        def __init__(self, depth=12):
            super(IFormer, self).__init__()
            self.temp = nn.Parameter(0.07 * torch.ones(1), requires_grad=True)

        def forward(self, Former, text, image):
            text, image = Former(text, image)
            sim_q2t = torch.matmul(image, text.transpose(1, 2))
            sim_t2q = torch.matmul(text, image.transpose(1, 2))
            img2txt, _ = torch.max(sim_q2t, dim=-1)
            txt2img, _ = torch.max(sim_t2q, dim=-1)
            img2txt = img2txt / self.temp
            txt2img = txt2img / self.temp

            loss = 0
            for i in range(img2txt.shape[0]):
                targets = torch.arange(0, img2txt.shape[1], dtype=torch.bfloat16).to(device) # 0, 1, 2, ..., 63
                loss_itc = (CrossEntropyLoss(label_smoothing=0.1)(img2txt[i], targets) + CrossEntropyLoss(label_smoothing=0.1)(txt2img[i], targets)) / 2
                loss += loss_itc
            loss /= img2txt.shape[0]

            return loss


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    Net_Prefix = Prefix().to(device).to(torch.bfloat16)
    Net_IFormer = IFormer().to(device).to(torch.bfloat16)
    optimizer_F = optim.Adam(Net_IFormer.parameters(), lr=optimizer_F_lr)

    train_losses_F = []
    test_losses_F = []
    save = []
    present_epoch = 1
    best_train_loss_F = 9999999999
    best_test_loss_F = 9999999999


    # if checkpoint:
    #     checkpoint_F = torch.load('./Model/' + load_name + "/" + load_name + '_NetIFormer_' + str(load_num) + '.pth')
    #     Net_Former.load_state_dict(checkpoint_F['model_state_dict'])
    #     optimizer_F.load_state_dict(checkpoint_F['optimizer_state_dict'])
    #     present_epoch = checkpoint_F['epoch'] + 1
    #     del checkpoint_F
    #     gc.collect()

    startTime = 0
    textExtractionTime = 0
    GeneratorForwardTime = 0
    GeneratorBackwardTime = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        print("---------------------------------------- epoch " + str(
            epoch + present_epoch) + " ---------------------------------------")
        train_loss_F = 0
        test_loss_F = 0
        pre = 0
        ###################################### Train ######################################
        with tqdm(train_loader, unit="batch", leave=True) as tepoch:
            for idx, (text, image, funny_score) in enumerate(tepoch):
                # tepoch.set_postfix({'Now': tepoch.format_dict['elapsed'], 'Status': " New batch preprocessing"})
                if idx == 0:
                    startTime = tepoch.format_dict['elapsed']
                elif idx == 1:
                    startTime = (tepoch.format_dict['elapsed'] - pre) * 2
                else:
                    startTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']
                text = textExtraction_IFT(tokenizer, config, text)
                image = image.to(torch.bfloat16)
                textExtractionTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']
                ######################################################
                # (1) Update Generator network
                ######################################################
                optimizer_F.zero_grad()
                loss_F = Net_IFormer(Net_Prefix, text.to(device), image.to(device))
                train_loss_F += loss_F.item()
                GeneratorForwardTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']
                loss_F.backward()
                optimizer_F.step()
                GeneratorBackwardTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']
                ######################################################
                # tepoch.set_postfix({'FC_loss': train_loss_FC/ (idx+1), 'G_loss': train_loss_G/ (idx+1), 'D_loss': train_loss_D/ (idx+1)})
                tepoch.set_postfix({'start': startTime / (idx + 1), 'textExtraction': textExtractionTime / (idx + 1),
                                    'GeneratorForward': GeneratorForwardTime / (idx + 1),
                                    'GeneratorBackwardG': GeneratorBackwardTime / (idx + 1)})
                ######################################################
        train_loss_F /= len(train_loader)
        train_losses_F.append(train_loss_F)
        ###################################### Train ######################################

        ######################################  Test ######################################
        with tqdm(test_loader, unit="batch", leave=True) as tepoch:
            for idx, (text, image, funny_score) in enumerate(tepoch):
                text = textExtraction_IFT(tokenizer, config, text)
                image = image.to(torch.bfloat16)
                # Generator
                loss_F = Net_IFormer(Net_Prefix, text.to(device), image.to(device))
                test_loss_F += loss_F.item()
                tepoch.set_postfix({'test_loss_F': test_loss_F / (idx + 1)})
        test_loss_F /= len(test_loader)
        test_losses_F.append(test_loss_F)
        ######################################  Test ######################################

        ######################################  Save ######################################
        hasSaved = False
        # 任一個loss小於最佳loss就存檔
        if best_train_loss_F > train_loss_F and best_test_loss_F > test_loss_F:
            best_train_loss_F = train_loss_F
            best_test_loss_F = test_loss_F
            torch.save({
                'epoch': epoch + present_epoch,
                'model_state_dict': Net_Prefix.state_dict(),
                'optimizer_state_dict': optimizer_F.state_dict(),
                'loss': train_loss_F,
            }, './Model/' + save_name + "/" + save_name + '_NetPrefix_' + str(epoch + present_epoch) + '.pth')
            hasSaved = True

        if hasSaved:
            save.append("V")
        else:
            save.append(" ")

        loss_data = pd.DataFrame()
        loss_data['train_F'] = train_losses_F
        loss_data['test_F'] = test_losses_F
        loss_data['save'] = save
        loss_data.to_csv('./Model/' + save_name + "/" + save_name + '_loss.csv', index=False)

        plt.figure()
        plt.plot(train_losses_F, label='train_F')
        plt.plot(test_losses_F, label='test_F')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        # save plot
        plt.savefig('./Model/' + save_name + "/" + save_name + '_loss.png')
        ######################################  Save ######################################

if __name__ == '__main__':
    train()