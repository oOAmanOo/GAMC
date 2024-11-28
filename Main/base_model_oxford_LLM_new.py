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
    batch_size = 8
    optimizer_Former_lr = 1e-5
    save_name = 'test'
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    ### 官方的Gemma #########################################################################################
    # 2b = 2304, 9b = 3584, 27b = 4608
    gemma_hiddenstate_size = 2304
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    gemmaConfig = AutoConfig.from_pretrained('google/gemma-2-2b-it')
    ### gemma float32 / bfloat16
    gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16)
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
            multi_out = self.multiheadAttentionLinear1(multi_out)
            multi_out = self.multiheadAttentionRelu(multi_out)
            multi_out = self.multiheadAttentionLinear2(multi_out)
            multi_out = self.multiheadAttentionLayerNorm(multi_out + text)

            prefix = self.prefix_const.unsqueeze(0).expand(image.shape[1], -1, -1).transpose(0, 1).to(device).to(
                torch.bfloat16)
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

            return self_out, multi_out


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

    class Generator(nn.Module):
        def __init__(self, depth=12):
            super(Generator, self).__init__()
            # gemma
            self.gemmaLinearMaxTokens = nn.Linear(64, 32)
            self.gemmaLinearBefore = nn.Linear(768, gemmaConfig.vocab_size)
            self.gemmaSoftmax = nn.Softmax(dim=2)
            # feed forward
            self.feedForwardLinear = nn.Linear(768, 768)
            self.feedForwardLayerNorm = nn.LayerNorm(768, eps=eps)
            # funny score
            self.FunnyScorelinear1 = nn.Linear(768, 1)
            self.FunnyScorelinear2 = nn.Linear(64, 1)
            # conditional/unconditional
            self.con_mlp1 = nn.Linear(1536, 2)
            self.con_mlp2 = nn.Linear(64, 1)
            self.unc_mlp1 = nn.Linear(768, 1)
            self.unc_mlp2 = nn.Linear(64, 1)

        def forward(self, Former_G, Former_F, image, text, train):
            ##############################################   generate ##############################################
            image_G, text_G = Former_G(text, image)
            ####################### gemma  generate #######################
            image_G = image_G.transpose(0, 1)
            image_G = self.gemmaLinearMaxTokens(image_G.transpose(1, 2)).transpose(1, 2)
            image_G = self.gemmaLinearBefore(image_G)
            image_G = self.gemmaSoftmax(image_G + eps)
            # get max value of each row, total 32*64
            top_k_values, top_k_indices = torch.topk(image_G, 1, dim=2, largest=True)
            output_text = textExtractReverse(gemma, tokenizer, gemmaConfig, top_k_indices).to(device).to(torch.bfloat16)
            g_C_g = torch.cat((output_text, image), dim=-1)
            ########################  conditional  ########################
            g_C_g = self.con_mlp1(g_C_g)
            g_C_g = self.con_mlp2(g_C_g.transpose(1, 2)).squeeze(-1)
            ######################## unconditional ########################
            g_UC_g = self.unc_mlp1(output_text).squeeze(-1)
            g_UC_g = self.unc_mlp2(g_UC_g).squeeze(-1)
            #######################################################################################################

            ###########################################   funny score   ###########################################
            image_F, text_F = Former_F(output_text, image)
            ######################### funny score #########################
            feature_fusion = image_F + text_F  # visual_attending_textual + textual_attending_visual
            feature_fusionFF = self.feedForwardLinear(feature_fusion)
            feature_fusion_final = self.feedForwardLayerNorm(feature_fusion + feature_fusionFF)
            feature_fusion_final = feature_fusion_final.squeeze(-1)
            feature_fusion_final = feature_fusion_final.transpose(0, 1)
            output_funny_score = self.FunnyScorelinear1(feature_fusion_final).squeeze(-1)
            output_funny_score = self.FunnyScorelinear2(output_funny_score).squeeze(-1)
            #######################################################################################################

            return g_C_g, g_UC_g, output_funny_score


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    NetFormer_G = Former().to(torch.bfloat16).to(device)
    NetFormer_F = Former().to(torch.bfloat16).to(device)
    Generator = Generator().to(torch.bfloat16).to(device)
    optimizer_Former = optim.Adam(Generator.parameters(), lr=optimizer_Former_lr)

    train_losses_Former = []
    test_losses_Former = []
    save = []
    present_epoch = 1
    best_train_loss_Former = 9999999999
    best_test_loss_Former = 9999999999
    g_con_loss_list = []
    g_unc_loss_list = []
    g_fc_loss_list = []

    checkpoint_Former = torch.load('.//Model/20241127_blipLoss_wo_co_attention_temp_NetFormer_41.pth')
    NetFormer_G.load_state_dict(checkpoint_Former['model_state_dict'])

    class BypassMultiheadAttention(nn.Module):
        def forward(self, query, key, value, attn_mask=None, *args, **kwargs):
            # 直接返回輸入 query，模擬無操作的情況
            return query, None

    for layers in NetFormer_G.layers_self_multi:
        layers.multiheadAttentionMultihead = BypassMultiheadAttention()
        layers.multiheadAttentionLinear1 = nn.Identity()
        layers.multiheadAttentionRelu = nn.Identity()
        layers.multiheadAttentionLinear2 = nn.Identity()
        layers.multiheadAttentionLayerNorm = nn.Identity()
    NetFormer_F.load_state_dict(checkpoint_Former['model_state_dict'])
    del checkpoint_Former
    gc.collect()

    def loss_function(condition_logits, uncondition_logits, funny_score, target_funny_score):
        result_fake_con = torch.ones(condition_logits.shape[0]).to(torch.long).to(device)  # 111 share weight
        result_fake_unc = torch.FloatTensor(condition_logits.shape[0]).to(torch.bfloat16).uniform_(0.9, 1.0).to(device)
        con_loss = CrossEntropyLoss(label_smoothing=0.1)(condition_logits, result_fake_con)
        unc_loss = BCEWithLogitsLoss()(uncondition_logits, result_fake_unc)

        fc_loss = nn.MSELoss()(funny_score, target_funny_score)

        g_con_loss_list.append(con_loss.item())
        g_unc_loss_list.append(unc_loss.item())
        g_fc_loss_list.append(fc_loss.item())
        loss = con_loss + unc_loss + fc_loss

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
                text = textExtraction(tokenizer, gemmaConfig, text).to(torch.bfloat16)
                image = image.to(torch.bfloat16)
                funny_score = funny_score.to(torch.bfloat16)
                optimizer_Former.zero_grad()
                con_logits, unc_logits, output_funny_score = Generator(NetFormer_G, NetFormer_F, text.to(device), image.to(device), funny_score.to(device))
                loss = loss_function(con_logits, unc_logits, output_funny_score, funny_score.to(device))
                loss.backward()
                optimizer_Former.step()
                train_loss_Former += loss.item()
                tepoch.set_postfix(loss=train_loss_Former / (idx + 1))
                ##########################################################################
                seperateLoss_data = pd.DataFrame()
                seperateLoss_data['con_loss'] = g_con_loss_list
                seperateLoss_data['unc_loss'] = g_unc_loss_list
                seperateLoss_data['fc_loss'] = g_fc_loss_list
                seperateLoss_data.to_csv('./Model/' + save_name + '/' + save_name + '_seperateLoss.csv', index=False)
                ##########################################################################
        train_loss_Former /= len(train_loader)
        train_losses_Former.append(train_loss_Former)
        ###################################### Test ######################################
        with tqdm(test_loader, unit="batch", leave=True) as tepoch:
            for idx, (text, image, funny_score) in enumerate(tepoch):
                text = textExtraction(tokenizer, gemmaConfig, text).to(torch.bfloat16)
                image = image.to(torch.bfloat16)
                funny_score = funny_score.to(torch.bfloat16)
                con_logits, unc_logits, output_funny_score = Generator(NetFormer_G, NetFormer_F, text.to(device), image.to(device), funny_score.to(device))
                loss = loss_function(con_logits, unc_logits, output_funny_score, funny_score.to(device))
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