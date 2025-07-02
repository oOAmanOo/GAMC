import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from Main.extractor import textExtraction, imageExtraction
eps = torch.finfo(torch.bfloat16).eps

batch_size = 15
dirPath = '../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
data = pd.read_csv(dirPath)
print("shape of data: ", data.shape)
data = data.sample(n=2000, random_state=42, replace=True).reset_index(drop=True)
print("sample of data: ", data.shape)
train, test = train_test_split(data, test_size=0.2, random_state=42)
#### GPT2 #########################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt_embedding_size = gpt.transformer.wte.weight.shape[1]
# prefix = image embedding, token = text embedding
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

    def image(self, image):
        for self_layer in self.layers_self:
            image = self_layer(image)
        return image

    def text(self, text):
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
    def __init__(self, Former, gpt, depth=12):
        super(Generator, self).__init__()
        self.Former = Former
        self.gpt = gpt
        # # self.gpt.eval()
        # self.gptLinearBefore = nn.Linear(768, gpt_embedding_size)
        # # feed forward
        # self.feedForwardLinear = nn.Linear(768, 768)
        # self.feedForwardLayerNorm = nn.LayerNorm(768, eps=eps)
        self.Only_image = None
        self.Only_image_text = []
        self.Only_image_loss = []
        self.Empty_Text = None
        self.Empty_Text_text = []
        self.Empty_Text_loss = []
        self.Full_Text = None
        self.Full_Text_text = []
        self.Full_Text_loss = []
        
    def forward(self, image, text_id, mode):
        if mode == 'test':
            dummy = torch.full_like(text_id, 50256, dtype=torch.long).to(device)
            text_embedd = gpt.transformer.wte(dummy)
        else:
            text_embedd = gpt.transformer.wte(text_id)
        ##############################################   generate ##############################################
        image = image.transpose(0, 1)
        image_G = self.Former.image(image)
        ########################################### input embedd ###########################################
        ########################################### feature fusion ###########################################
        text_embedd = text_embedd.transpose(0, 1)
        text_G = self.Former.text(text_embedd)
        image_G = image_G.transpose(0, 1)
        text_G = text_G.transpose(0, 1)
        # feature_fusion = image_G + text_G  # visual_attending_textual + textual_attending_visual
        # feature_fusionFF = self.feedForwardLinear(feature_fusion)
        # feature_fusion_final = self.feedForwardLayerNorm(feature_fusion + feature_fusionFF)
        embedding_cat = torch.cat((image_G, text_G), dim=1)
        if mode == 'train':
            # dummy_token = torch.full((text_id.shape[0], 10), 50256, dtype=text_id.dtype).to(device)
            dummy_token2 = torch.full_like(text_id, 50256, dtype=text_id.dtype).to(device)
            labels = torch.cat((dummy_token2, text_id), dim=1)
        else:
            labels = torch.full_like(embedding_cat[:, :, 0], 50256, dtype=text_id.dtype).to(device)
        # image_GG = self.gptLinearBefore(feature_fusion_final)
        gotOutput = self.gpt(inputs_embeds=embedding_cat, labels=labels)
        logits = gotOutput.logits
        if mode == 'test':
            # dummy_token = torch.full((text_id.shape[0], 10), 50256, dtype=text_id.dtype).to(device)
            dummy_token2 = torch.full_like(text_id, 50256, dtype=text_id.dtype).to(device)
            labels = torch.cat((dummy_token2, text_id), dim=1)
        return logits, labels

    def generate_woText(self, image, text_id):
        image = image.transpose(0, 1)
        image_G = self.Former.image(image)
        image_G = image_G.transpose(0, 1)
        gptOutput = self.gpt(inputs_embeds=image_G)
        logits = gptOutput.logits
        caption_tokens = logits.argmax(-1)[0]
        caption = tokenizer.decode(caption_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(caption)
        print(caption_tokens)
        loss = loss_function(logits, text_id)
        print("loss: ", loss.item())
        if self.Only_image != None:
            self.Only_image = torch.cat((self.Only_image, caption_tokens.unsqueeze(0)), dim=0)
        else:
            self.Only_image = caption_tokens.unsqueeze(0)
        self.Only_image_text.append(caption)
        self.Only_image_loss.append(loss.item())

    def generate_withText(self, image, text_id, mode):
        if mode == 'test':
            dummy = torch.full_like(text_id, 50256, dtype=text_id.dtype).to(device)
            text_embedd = self.gpt.transformer.wte(dummy)
        else:
            text_embedd = self.gpt.transformer.wte(text_id)
        ##############################################   generate ##############################################
        image = image.transpose(0, 1)
        image_G = self.Former.image(image)
        ########################################### feature fusion ###########################################
        text_embedd = text_embedd.transpose(0, 1)
        text_G = self.Former.text(text_embedd)
        image_G = image_G.transpose(0, 1)
        text_G = text_G.transpose(0, 1)
        embedding_cat = torch.cat((image_G, text_G), dim=1)
        # feature_fusion = image_G + text_G  # visual_attending_textual + textual_attending_visual
        # feature_fusionFF = self.feedForwardLinear(feature_fusion)
        # feature_fusion_final = self.feedForwardLayerNorm(feature_fusion + feature_fusionFF)
        # image_GG = self.gptLinearBefore(feature_fusion_final)
        gptOutput = self.gpt(inputs_embeds=embedding_cat)
        logits = gptOutput.logits
        caption_tokens = logits.argmax(-1)[0]
        caption = tokenizer.decode(caption_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(caption)
        print(caption_tokens)
        # dummy_token = torch.full((text_id.shape[0], 10), 50256, dtype=text_id.dtype).to(device)
        dummy_token2 = torch.full_like(text_id, 50256, dtype=text_id.dtype).to(device)
        labels = torch.cat((text_id, text_id), dim=1)
        loss = loss_function(logits, labels)
        print("loss: ", loss.item())
        if mode == 'test':
            if self.Empty_Text != None:
                self.Empty_Text = torch.cat((self.Empty_Text, caption_tokens.unsqueeze(0)), dim=0)
            else:
                self.Empty_Text = caption_tokens.unsqueeze(0)
            self.Empty_Text_text.append(caption)
            self.Empty_Text_loss.append(loss.item())
        else:
            if self.Full_Text != None:
                self.Full_Text = torch.cat((self.Full_Text, caption_tokens.unsqueeze(0)), dim=0)
            else:
                self.Full_Text = caption_tokens.unsqueeze(0)
            self.Full_Text_text.append(caption)
            self.Full_Text_loss.append(loss.item())
            
    def print(self):
        def dataframe_Name(name, count=128, rows=1):
            if rows == 1:
                name_df = pd.DataFrame([[name]], columns=["Name"])
                num_df = pd.DataFrame([[0] * count for _ in range(rows)], columns=[f"{i}" for i in range(0, count)])
                name_df = pd.concat([name_df, num_df], axis=1)
                return name_df
            else:
                name_df = pd.DataFrame([[name] * count for _ in range(rows)], columns=[f"{i}" for i in range(64, 128)])
                return name_df

        test = pd.DataFrame()
        test['Name'] = ['test1', 'test2', 'train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7', 'train8', 'train9',
                        'train10']
        Only_image_df = pd.DataFrame(self.Only_image.cpu().detach(), columns=[f"{i}" for i in range(0, 64)])
        Only_image_df = pd.concat([test, Only_image_df, dataframe_Name("-", 64, 10)], axis=1)
        Only_image_df['loss'] = self.Only_image_loss
        Only_image_df['text'] = self.Only_image_text
        Empty_Text_df = pd.DataFrame(self.Empty_Text.cpu().detach(), columns=[f"{i}" for i in range(0, 128)])
        Empty_Text_df = pd.concat([test, Empty_Text_df], axis=1)
        Empty_Text_df['loss'] = self.Empty_Text_loss
        Empty_Text_df['text'] = self.Empty_Text_text
        Full_Text_df = pd.DataFrame(self.Full_Text.cpu().detach(), columns=[f"{i}" for i in range(0, 128)])
        Full_Text_df = pd.concat([test, Full_Text_df], axis=1)
        Full_Text_df['loss'] = self.Full_Text_loss
        Full_Text_df['text'] = self.Full_Text_text
        final = pd.concat([dataframe_Name("Only_image"), Only_image_df, dataframe_Name("Empty_Text"), Empty_Text_df,
                           dataframe_Name("Full_Text"), Full_Text_df], axis=0)
        print(dataframe_Name("Only_image").shape, Only_image_df.shape, dataframe_Name("Empty_Text").shape,
              Empty_Text_df.shape, dataframe_Name("Full_Text").shape, Full_Text_df.shape)
        print(final.shape)
        print(final.columns)
        final.to_csv('./Model/' + checkpoint_folder + '/' + checkpoint_folder + '_test.csv', index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############# load  model #############
NetFormer = Former().to(torch.bfloat16).to(device)
Generator = Generator(Former=NetFormer, gpt=gpt).to(torch.bfloat16).to(device)
#######################################
checkpoint_folder = '20241219_Clip_GPT_catFF_64'
checkpoint_Generator = './Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetLLM_7.pth'
#######################################
checkpoint_Generator = torch.load(checkpoint_Generator)
Generator.load_state_dict(checkpoint_Generator['model_state_dict'])
#######################################

def loss_function(logits, labels):
    logits = logits.contiguous().view(-1, logits.size(-1))
    labels = labels.contiguous().view(-1)
    loss = nn.CrossEntropyLoss(ignore_index=50256)(logits, labels)
    return loss

# generate
Generator.eval()

print('=======================  test  1 =========================')
image = imageExtraction("./test_img.jpg").to(device, dtype=torch.bfloat16)
test_gt = ['you all loved it so i brought it back']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  test  2 =========================')
image = imageExtraction("./test_img2.jpg").to(device, dtype=torch.bfloat16)
test_gt = ['Can’t believe this is real (still) life. The Saucy Nugg legacy is forever cemented in oil on canvas (2024). Thank you']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')




print('=======================  sample1 =========================')
print(train[train['image_id'] == 'bokete_51227'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/imgflip_34.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['You finish doing something at your friends house and look at your phone; 7 missed calls from your mom; 7 missed calls from your mom']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  sample2 =========================')
print(train[train['image_id'] == 'bokete_3820'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/bokete_3820.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['I\'m in my 50s!']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  sample3 =========================')
print(train[train['image_id'] == 'imgflip_0'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/imgflip_0.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['Getting hurt from cutting open your leg; Getting hurt from taking off a bandaid']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  sample4 =========================')
print(train[train['image_id'] == 'imgflip_8'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/imgflip_8.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['image tagged in memes,one does not simply']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  sample5 =========================')
print(train[train['image_id'] == 'imgflip_15'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/imgflip_15.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['KIDS WHEN THEIR PARENTS GIVE THEM; "THE TALK"']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  sample6 =========================')
print(train[train['image_id'] == 'imgflip_19'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/imgflip_19.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['NOT SURE IF PEOPLE ARE UPVOTING MEMES; OR USER NAMES']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  sample7 =========================')
print(train[train['image_id'] == 'bokete_104530'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/bokete_104530.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['It\'s a family night runaway.']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  sample8 =========================')
print(train[train['image_id'] == 'imgflip_730'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/imgflip_730.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['CHUCK IS THE GOOD TYPE OF SCUMBAG; CUZ HE ONLY ROASTS YOU FROM YOUR INSIDES']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  sample9 =========================')
print(train[train['image_id'] == 'imgflip_130'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/imgflip_130.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['SO YOUR TELLIN\' ME THAT SCHOOLS GOOD FOR YOU']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

print('=======================  sample10 =========================')
print(train[train['image_id'] == 'imgflip_677'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageData/imgflip_677.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['Y\'ALL GOT ANY MORE OF THEM; JOBS?']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
Generator.generate_withText(image, text_id,'train')

Generator.print()
