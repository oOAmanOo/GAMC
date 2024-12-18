import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
eps = torch.finfo(torch.bfloat16).eps

batch_size = 15
dirPath = '../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
data = pd.read_csv(dirPath)
print("shape of data: ", data.shape)
data = data.sample(n=10000, random_state=42, replace=True).reset_index(drop=True)
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

Only_image = None
Empty_Text = None
Full_Text = None
Only_image_text = []
Empty_Text_text = []
Full_Text_text = []
class MLP(nn.Module):
    def __init__(self, sizes: tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
class MlpTransformer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(gpt_embedding_size, h_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h_dim, gpt_embedding_size)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class TransformerLayer(nn.Module):
    def __init__(self,  mlp_ratio=4., bias=False, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(gpt_embedding_size, eps=eps)
        self.attn = nn.MultiheadAttention(gpt_embedding_size, 8, bias=bias, dropout=dropout)
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
            layers.append(TransformerLayer(mlp_ratio))
        self.layers = nn.ModuleList(layers)
    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions
    def forward(self, x, y=None, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], x.shape[2]).to(device).to(torch.bfloat16)
        for i, layer in enumerate(self.layers):
            x = layer(x, x, mask)
            # if i % 2 == 0 and self.enc_dec:  # cross
            #     x = layer(x, y)
            # elif self.enc_dec:  # self
            #     x = layer(x, x, mask)
            # else:  # self or cross
            #     x = layer(x, y, mask)
        return x
class TransformerMapper(nn.Module):
    def __init__(self):
        super(TransformerMapper, self).__init__()
        self.transformer = Transformer(8, enc_dec=True)
        self.linear = nn.Linear(512, 64*gpt_embedding_size)
        self.prefix_const = nn.Parameter(torch.randn(64, gpt_embedding_size), requires_grad=True)
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], 64, gpt_embedding_size)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], -1, -1).to(device).to(torch.bfloat16)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)
        return out
class Generator(nn.Module):
    def __init__(self, Former, gpt, depth=12):
        super(Generator, self).__init__()
        if Former == "MLP":
            self.Former = MLP()
        else:
            self.Former = TransformerMapper()
        self.gpt = gpt
        self.gpt.train()
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
            dummy = torch.full((image.shape[0], 64), 50256, dtype=torch.long).to(device)
            text_embedd = gpt.transformer.wte(dummy)
        else:
            text_embedd = gpt.transformer.wte(text_id)
        ##############################################   generate ##############################################
        image_G = self.Former(image)
        embedding_cat = torch.cat((image_G, text_embedd), dim=1)
        ########################################### feature fusion ###########################################
        if mode == 'train':
            dummy_token = torch.full((text_id.shape[0], 64), 50256, dtype=text_id.dtype).to(device)
            dummy_token2 = torch.full_like(text_id, 50256, dtype=text_id.dtype).to(device)
            labels = torch.cat((dummy_token, text_id, dummy_token2), dim=1)
        else:
            labels = torch.full_like(embedding_cat[:, :, 0], 50256, dtype=text_id.dtype).to(device)
        gptOutput = self.gpt(inputs_embeds=embedding_cat, labels=labels)
        logits = gptOutput.logits
        if mode == 'test':
            dummy_token = torch.full((text_id.shape[0], 64), 50256, dtype=text_id.dtype).to(device)
            dummy_token2 = torch.full_like(text_id, 50256, dtype=text_id.dtype).to(device)
            labels = torch.cat((dummy_token, text_id, dummy_token2), dim=1)
        return logits, labels

    def generate_woText(self, image, text_id):
        image_GG = self.Former(image)
        gptOutput = self.gpt(inputs_embeds=image_GG)
        logits = gptOutput.logits
        caption_tokens = logits.argmax(-1)[0]
        caption = tokenizer.decode(caption_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(caption)
        print(caption_tokens)
        dummy_token = torch.full((text_id.shape[0], 64), 50256, dtype=text_id.dtype).to(device)
        labels = torch.cat((dummy_token, text_id), dim=1)
        loss = loss_function(logits, labels)
        print("loss: ", loss.item())
        if self.Only_image != None:
            self.Only_image = torch.cat((self.Only_image, caption_tokens.unsqueeze(0)), dim=0)
        else:
            self.Only_image = caption_tokens.unsqueeze(0)
        self.Only_image_text.append(caption)
        self.Only_image_loss.append(loss.item())
        
    def generate_withText(self, image, text_id, trainTest):
        if trainTest == 'test':
            dummy = torch.full_like(text_id, 50256, dtype=text_id.dtype).to(device)
            text_embedd = self.gpt.transformer.wte(dummy)
        else:
            text_embedd = self.gpt.transformer.wte(text_id)
        ##############################################   generate ##############################################
        image_G = self.Former(image)
        ########################################### feature fusion ###########################################
        image_GG = torch.cat((image_G, text_embedd), dim=1)
        gptOutput = self.gpt(inputs_embeds=image_GG)
        logits = gptOutput.logits
        caption_tokens = logits.argmax(-1)[0]
        caption = tokenizer.decode(caption_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(caption)
        print(caption_tokens)
        dummy_token = torch.full((text_id.shape[0], 64), 50256, dtype=text_id.dtype).to(device)
        dummy_token2 = torch.full_like(text_id, 50256, dtype=text_id.dtype).to(device)
        labels = torch.cat((dummy_token, text_id, dummy_token2), dim=1)
        print(logits.shape, labels.shape)
        loss = loss_function(logits, labels)
        print("loss: ", loss.item())
        if trainTest == 'test':
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
        def dataframe_Name(name,count=0, rows=0):
            if count == 0:
                name_df = pd.DataFrame([[name]], columns=["Name"])
                return name_df
            else:
                name_df = pd.DataFrame([[name] * count for _ in range(rows)], columns=[f"{i}" for i in range(0, count)])
                return name_df

        test = pd.DataFrame()
        test['Name'] = ['train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7', 'train8', 'train9',
                        'train10']
        Only_image_df = pd.DataFrame(self.Only_image.cpu().detach(), columns=[f"{i}" for i in range(64, 192)])
        Only_image_df = pd.concat([test, Only_image_df, dataframe_Name("-", 64, 10)], axis=1)
        Only_image_df['loss'] = self.Only_image_loss
        Only_image_df['text'] = self.Only_image_text
        Empty_Text_df = pd.DataFrame(self.Empty_Text.cpu().detach(), columns=[f"{i}" for i in range(0, 192)])
        Empty_Text_df = pd.concat([test, Empty_Text_df], axis=1)
        Empty_Text_df['loss'] = self.Empty_Text_loss
        Empty_Text_df['text'] = self.Empty_Text_text
        Full_Text_df = pd.DataFrame(self.Full_Text.cpu().detach(), columns=[f"{i}" for i in range(0, 192)])
        Full_Text_df = pd.concat([test, Full_Text_df], axis=1)
        Full_Text_df['loss'] = self.Full_Text_loss
        Full_Text_df['text'] = self.Full_Text_text
        final = pd.concat([dataframe_Name("Only_image"), Only_image_df, dataframe_Name("Empty_Text"), Empty_Text_df,
                           dataframe_Name("Full_Text"), Full_Text_df], axis=0)
        print(dataframe_Name("Only_image").shape, Only_image_df.shape, dataframe_Name("Empty_Text").shape, Empty_Text_df.shape, dataframe_Name("Full_Text").shape, Full_Text_df.shape)
        print(final.shape)
        print(final.columns)
        final.to_csv('./Model/' + checkpoint_folder + '/' + checkpoint_folder + '_test.csv', index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############# load  model #############
# NetFormer = Former().to(torch.bfloat16).to(device)
Generator = Generator(Former= "Trans", gpt=gpt).to(torch.bfloat16).to(device)
#######################################
checkpoint_folder = '20241218_Clip_Clip_Clip_noGPT_testNoText_img2txtonly'
checkpoint_Generator = './Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetLLM_72.pth'
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

# print('=======================  test  1 =========================')
# image = imageExtraction("./test_img.jpg").to(device, dtype=torch.bfloat16)
# test_gt = ['you all loved it so i brought it back']
# print("ground_truth: ", test_gt)
# text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
# text_id = text_id['input_ids'].to(device)
# print("========= Only image =========")
# Generator.generate_woText(image)
# print("========= Empty Text =========")
# Generator.generate_withText(image)
# print("========= Full  Text =========")
# Generator.generate_withText(image, text_id)
#
# print('=======================  test  2 =========================')
# image = imageExtraction("./test_img2.jpg").to(device, dtype=torch.bfloat16)
# test_gt = ['Can’t believe this is real (still) life. The Saucy Nugg legacy is forever cemented in oil on canvas (2024). Thank you']
# print("ground_truth: ", test_gt)
# text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
# text_id = text_id['input_ids'].to(device)
# print("========= Only image =========")
# Generator.generate_woText(image)
# print("========= Empty Text =========")
# Generator.generate_withText(image)
# print("========= Full  Text =========")
# Generator.generate_withText(image, text_id)



print('=======================  sample1 =========================')
print(train[train['image_id'] == 'bokete_51227'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_34.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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
image = torch.load('../../Oxford_HIC/ImageClip/bokete_3820.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_0.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_8.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_15.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_19.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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
image = torch.load('../../Oxford_HIC/ImageClip/bokete_104530.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_730.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_130.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_677.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
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


