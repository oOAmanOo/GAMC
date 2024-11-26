import os
import gc
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from local_gemma import LocalGemma2ForCausalLM

from extractor import addImagePath, textExtraction, imageExtraction, textExtractReverse
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
    epochs = 30
    batch_size = 128
    optimizer_Former_lr = 1e-5
    save_name = '20241125_old_IFT_blipLoss_wo_co_attention_shareWeight'
    if not os.path.exists('./Model/' + save_name):
        os.makedirs('./Model/' + save_name)
        os.makedirs('D:/MemeGAN/Model/' + save_name)


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)

    ### 官方的Gemma #########################################################################################
    # 2b = 2304, 9b = 3584, 27b = 4608
    # gemma_hiddenstate_size = 2304
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    gemmaConfig = AutoConfig.from_pretrained('google/gemma-2-2b-it')
    ### gemma float32 / bfloat16
    # gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16)
    ########################################################################################################

    class self_multi(nn.Module):
        def __init__(self):
            super(self_multi, self).__init__()
            # self attention
            self.selfAttentionMultihead = nn.MultiheadAttention(768, 1)
            self.selfAttentionLayerNorm = nn.LayerNorm(768, eps=eps)
            self.selfAttentionLinear1 = nn.Linear(768, 768)
            self.selfAttentionRelu = nn.ReLU()
            self.selfAttentionLinear2 = nn.Linear(768, 768)
            self.selfAttentionLayerNorm2 = nn.LayerNorm(768, eps=eps)

            # multihead attention
            self.multiheadAttentionMultihead = nn.MultiheadAttention(768, 8)
            self.multiheadAttentionLinear1 = nn.Linear(768, 768)
            self.multiheadAttentionRelu = nn.ReLU()
            self.multiheadAttentionLinear2 = nn.Linear(768, 768)
            self.multiheadAttentionLayerNorm = nn.LayerNorm(768, eps=eps)

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

        def forward(self, image, text):
            # self attention module
            self_temp = self.selfAttentionMultihead(image, image, image)[0]
            self_temp = self.selfAttentionLayerNorm(self_temp + image)
            self_out = self.selfAttentionLinear1(self_temp)
            self_out = self.selfAttentionRelu(self_out)
            self_out = self.selfAttentionLinear2(self_out)
            self_out = self.selfAttentionLayerNorm(self_out + self_temp)

            # multihead attention module
            multi_out = self.multiheadAttentionMultihead(text, text, text)[0]
            multi_out = self.selfAttentionLinear1(multi_out)
            multi_out = self.selfAttentionRelu(multi_out)
            multi_out = self.selfAttentionLinear2(multi_out)
            multi_out = self.selfAttentionLayerNorm(multi_out + text)

            # prefix = self.prefix_const.unsqueeze(0).expand(image.shape[1], -1, -1).transpose(0, 1).to(device).to(
            #     torch.bfloat16)
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

            return self_out, multi_out

    # class co_attention(nn.Module):
    #     def __init__(self):
    #         super(co_attention, self).__init__()
    #         self.prefix_const = nn.Parameter(torch.randn(64, 768), requires_grad=True)
    #         # co-attention text
    #         self.coAttentionTextMultihead = nn.MultiheadAttention(768, 8)
    #         self.coAttentionTextLinear1 = nn.Linear(768, 768)
    #         self.coAttentionTextRelu = nn.ReLU()
    #         self.coAttentionTextLinear2 = nn.Linear(768, 768)
    #         self.coAttentionTextLayerNorm = nn.LayerNorm(768, eps=eps)
    #
    #         # co-attention image
    #         self.coAttentionImageMultihead = nn.MultiheadAttention(768, 8)
    #         self.coAttentionImageLinear1 = nn.Linear(768, 768)
    #         self.coAttentionImageRelu = nn.ReLU()
    #         self.coAttentionImageLinear2 = nn.Linear(768, 768)
    #         self.coAttentionImageLayerNorm = nn.LayerNorm(768, eps=eps)
    #
    #         # feed forward
    #         self.feedForwardLinear1 = nn.Linear(768, 768)
    #         self.feedForwardRelu = nn.ReLU()
    #         self.feedForwardLinear2 = nn.Linear(768, 768)
    #
    #     def forward(self, image):
    #         prefix = self.prefix_const.unsqueeze(0).expand(image.shape[1], -1, -1).transpose(0, 1).to(device).to(torch.bfloat16)
    #         # co-attention image module
    #         visual_attending_textual = self.coAttentionTextMultihead(image, prefix, prefix)[0]
    #         visual_attending_textual = self.coAttentionTextLinear1(visual_attending_textual)
    #         visual_attending_textual = self.coAttentionTextRelu(visual_attending_textual)
    #         visual_attending_textual = self.coAttentionTextLinear2(visual_attending_textual)
    #         visual_attending_textual = self.coAttentionTextLayerNorm(visual_attending_textual + image)
    #
    #         # co-attention text module
    #         textual_attending_visual = self.coAttentionTextMultihead(prefix, image, image)[0]
    #         textual_attending_visual = self.coAttentionTextLinear1(textual_attending_visual)
    #         textual_attending_visual = self.coAttentionTextRelu(textual_attending_visual)
    #         textual_attending_visual = self.coAttentionTextLinear2(textual_attending_visual)
    #         textual_attending_visual = self.coAttentionTextLayerNorm(textual_attending_visual + prefix)
    #
    #         output = self.feedForwardLinear1(visual_attending_textual + textual_attending_visual)
    #         output = self.feedForwardRelu(output)
    #         output = self.feedForwardLinear2(output)
    #
    #         return output

    class Former(nn.Module):
        def __init__(self, depth=12):
            super(Former, self).__init__()
            self.layers_self_multi = nn.ModuleList([self_multi() for _ in range(depth)])
            # self.layers_co_attention = nn.ModuleList([co_attention() for _ in range(depth)])

        def forward(self, text, image):
            # max_seq_len = max(text.shape[1], image.shape[1])
            # text = nn.functional.pad(text, (0, 0, 0, max_seq_len - text.shape[1]))
            # image = nn.functional.pad(image, (0, 0, 0, max_seq_len - image.shape[1]))
            text = text.transpose(0, 1)
            image = image.transpose(0, 1)

            ######################### Transformer #########################
            for self_multi_layer in self.layers_self_multi:
                image, text = self_multi_layer(image, text)
            # for co_attention_layer in self.layers_co_attention:
            #     image = co_attention_layer(image)
            ###############################################################
            return image, text

    class IFormer(nn.Module):
        def __init__(self, depth=12):
            super(IFormer, self).__init__()
            self.temp = nn.Parameter(0.07 * torch.ones([]))
        def forward(self, text, image):

            text = nn.functional.normalize(text.transpose(0, 1), p=2, dim=-1)
            image = nn.functional.normalize(image.transpose(0, 1), p=2, dim=-1)

            c_text = text.unsqueeze(2).expand(-1, -1, image.shape[1], -1).to(torch.bfloat16)
            c_image = image.unsqueeze(2).expand(-1, -1, text.shape[1], -1).to(torch.bfloat16)
            sim_q2t = torch.einsum('bijk,bjik->bij', c_image, c_text)
            sim_t2q = torch.einsum('bijk,bjik->bij', c_text, c_image)
            ################ BITA Loss ################
            # img2txt, _ = torch.max(sim_q2t, dim=-1)
            # txt2img, _ = torch.max(sim_t2q, dim=-1)
            # img2txt = img2txt / self.temp
            # txt2img = txt2img / self.temp

            # loss = 0
            # for i in range(img2txt.shape[0]):
            #     sim_targets = torch.zeros(img2txt.size()).to(image.device)
            #     sim_targets.fill_diagonal_(1)
            #     targets = torch.arange(0, img2txt.shape[1], dtype=torch.bfloat16).to(device)
            #     loss_itc = (CrossEntropyLoss(label_smoothing=0.1)(img2txt[i], targets) + CrossEntropyLoss(
            #         label_smoothing=0.1)(txt2img[i], targets)) / 2
            #     loss += loss_itc
            # loss /= img2txt.shape[0]

            ################ Blip Loss ################
            img2txt, img2txt_idx = torch.max(sim_q2t, dim=-1, keepdim=True)
            txt2img, txt2img_idx = torch.max(sim_t2q, dim=-1, keepdim=True)

            img2txt_target = torch.zeros_like(sim_q2t, dtype=torch.bfloat16)
            txt2img_target = torch.zeros_like(sim_t2q, dtype=torch.bfloat16)
            img2txt_target.scatter_(-1, img2txt_idx, 1)
            txt2img_target.scatter_(-1, txt2img_idx, 1)

            img2txt_loss = CrossEntropyLoss()(sim_q2t, img2txt_target)
            txt2img_loss = CrossEntropyLoss()(sim_t2q, txt2img_target)
            loss = (img2txt_loss + txt2img_loss) / 2
            return loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    NetFormer = Former().to(torch.bfloat16).to(device)
    NetIFormer = IFormer().to(torch.bfloat16).to(device)
    optimizer_Former = optim.Adam(NetFormer.parameters(), lr=optimizer_Former_lr)

    train_losses_Former = []
    test_losses_Former = []
    save = []
    present_epoch = 1
    best_train_loss_Former = 9999999999
    best_test_loss_Former = 9999999999

    checkpoint = False
    if checkpoint:
        checkpoint_Former = torch.load('D:/MemeGAN/Model/20241105_15wan_dOnly/20241105_15wan_dOnly_NetFormer_5.pth')
        NetFormer.load_state_dict(checkpoint_Former['model_state_dict'])
        optimizer_Former.load_state_dict(checkpoint_Former['optimizer_state_dict'])
        present_epoch = checkpoint_Former['epoch'] + 1
        del checkpoint_Former
        gc.collect()

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        print("---------------------------------------- epoch " + str(
            epoch + present_epoch) + " ---------------------------------------")
        train_loss_Former = 0
        test_loss_Former = 0
        ###################################### Train ######################################
        with tqdm(train_loader, unit="batch", leave=True) as tepoch:
            for idx, (text, image, funny_score) in enumerate(tepoch):
                text = textExtraction(tokenizer, gemmaConfig, text).to(torch.bfloat16)
                image = image.to(torch.bfloat16)
                optimizer_Former.zero_grad()
                image, text = NetFormer(text.to(device), image.to(device))
                loss = NetIFormer(text.to(device), image.to(device))
                loss.backward()
                optimizer_Former.step()
                train_loss_Former += loss.item()
                tepoch.set_postfix(loss=train_loss_Former / (idx + 1))
        train_loss_Former /= len(train_loader)
        train_losses_Former.append(train_loss_Former)
        ###################################### Test ######################################
        with tqdm(test_loader, unit="batch", leave=True) as tepoch:
            for idx, (text, image, funny_score) in enumerate(tepoch):
                text = textExtraction(tokenizer, gemmaConfig, text).to(torch.bfloat16)
                image = image.to(torch.bfloat16)
                image, text = NetFormer(text.to(device), image.to(device))
                loss = NetIFormer(text.to(device), image.to(device))
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
                'loss': test_loss_Former,
            }, './Model/' + save_name + '/' + save_name + '_NetFormer_' + str(epoch + present_epoch) + '.pth')
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