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
        print(dummy_token.shape, text_id.shape)
        labels = torch.cat((dummy_token, text_id, text_id), dim=1)
    else:
        labels = torch.full_like(embedding_cat[:, :, 0], 50256, dtype=text_id.dtype).to(device)
    print(embedding_cat.shape, labels.shape)
    gptOutput = self.gpt(inputs_embeds=embedding_cat, labels=labels)
    logits = gptOutput.logits
    if mode == 'test':
        dummy_token = torch.full((text_id.shape[0], 64), 50256, dtype=text_id.dtype).to(device)
        labels = torch.cat((dummy_token, text_id, text_id), dim=1)
    return logits, labels

def generate_woText(self, image):
    image_GG = self.Former(image)
    print("----generate_beam----")
    # print(generate_beam(Generator, tokenizer, embed=image_GG)[0])
    a = generate_beam(Generator, tokenizer, embed=image_GG)[0]
    print(a)
    print(tokenizer(a, return_tensors='pt'))
    print("------generate2------")
    # print(generate2(Generator, tokenizer, embed=image_GG))
    a = generate2(Generator, tokenizer, embed=image_GG)
    print(a)
    print(tokenizer(a, return_tensors='pt'))
    print("----Not  Generate----")
    gptOutput = self.gpt(inputs_embeds=image_GG)
    logits = gptOutput.logits
    print(tokenizer.decode(logits.argmax(-1)[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    print(logits.argmax(-1)[0])


def generate_withText(self, image, text_id=None):
    if text_id is None:
        text_id = torch.full((image.shape[0], 64), 50256, dtype=torch.long).to(device)
    text_embedd = gpt.transformer.wte(text_id)
    ##############################################   generate ##############################################
    image_G = self.Former(image)
    ########################################### feature fusion ###########################################
    image_GG = torch.cat((image_G, text_embedd), dim=1)
    print("----generate_beam----")
    # print(generate_beam(Generator, tokenizer, embed=image_GG)[0])
    a = generate_beam(Generator, tokenizer, embed=image_GG)[0]
    print(tokenizer(a, return_tensors='pt'))
    print("------generate2------")
    # print(generate2(Generator, tokenizer, embed=image_GG))
    a = generate2(Generator, tokenizer, embed=image_GG)
    print(tokenizer(a, return_tensors='pt'))
    print("----Not  Generate----")
    gptOutput = self.gpt(inputs_embeds=image_GG)
    logits = gptOutput.logits
    print(tokenizer.decode(logits.argmax(-1)[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    print(logits.argmax(-1)[0])

def generate_beam(
        model,
        tokenizer,
        beam_size: int = 5,
        prompt=None,
        embed=None,
        entry_length=67,
        temperature=1.0,
        stop_token: str = ".",
):

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
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
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
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
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


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.0,
        stop_token: str = ".",
):
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

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                    # nn.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            if tokens is not None:
                output_list = tokens.squeeze().cpu().numpy()
                output_text = tokenizer.decode(output_list)
                generated_list.append(output_text)
            else:
                generated_list.append("")

    return generated_list[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############# load  model #############
# NetFormer = Former().to(torch.bfloat16).to(device)
Generator = Generator(Former= "Trans", gpt=gpt).to(torch.bfloat16).to(device)
#######################################
checkpoint_folder = '20241218_Clip_Clip_Clip_noGPT_testNoText_img2txt'
checkpoint_Generator = './Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetLLM_15.pth'
#######################################
checkpoint_Generator = torch.load(checkpoint_Generator)
Generator.load_state_dict(checkpoint_Generator['model_state_dict'])
#######################################

def loss_function(logits, labels):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
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
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)

print('=======================  sample2 =========================')
print(train[train['image_id'] == 'bokete_3820'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/bokete_3820.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['I\'m in my 50s!']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)

print('=======================  sample3 =========================')
print(train[train['image_id'] == 'imgflip_0'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_0.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['Getting hurt from cutting open your leg; Getting hurt from taking off a bandaid']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)

print('=======================  sample4 =========================')
print(train[train['image_id'] == 'imgflip_8'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_8.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['image tagged in memes,one does not simply']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)

print('=======================  sample5 =========================')
print(train[train['image_id'] == 'imgflip_15'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_15.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['KIDS WHEN THEIR PARENTS GIVE THEM; "THE TALK"']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)

print('=======================  sample6 =========================')
print(train[train['image_id'] == 'imgflip_19'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_19.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['NOT SURE IF PEOPLE ARE UPVOTING MEMES; OR USER NAMES']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)

print('=======================  sample7 =========================')
print(train[train['image_id'] == 'bokete_104530'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/bokete_104530.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['It\'s a family night runaway.']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)

print('=======================  sample8 =========================')
print(train[train['image_id'] == 'imgflip_730'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_730.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['CHUCK IS THE GOOD TYPE OF SCUMBAG; CUZ HE ONLY ROASTS YOU FROM YOUR INSIDES']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)

print('=======================  sample9 =========================')
print(train[train['image_id'] == 'imgflip_130'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_130.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['SO YOUR TELLIN\' ME THAT SCHOOLS GOOD FOR YOU']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)

print('=======================  sample10 =========================')
print(train[train['image_id'] == 'imgflip_677'].shape[0] > 0)
image = torch.load('../../Oxford_HIC/ImageClip/imgflip_677.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['Y\'ALL GOT ANY MORE OF THEM; JOBS?']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image)
print("========= Empty Text =========")
Generator.generate_withText(image)
print("========= Full  Text =========")
Generator.generate_withText(image, text_id)


