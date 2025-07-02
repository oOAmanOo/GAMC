import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer, Gemma2ForCausalLM
from typing import Tuple, Optional, Union
eps = torch.finfo(torch.bfloat16).eps

prefix_length = 10
dirPath = '../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
data = pd.read_csv(dirPath)
print("shape of data: ", data.shape)
data = data.sample(n=10000, random_state=42, replace=True).reset_index(drop=True)
print("sample of data: ", data.shape)
train, test = train_test_split(data, test_size=0.2, random_state=42)
### GPT2 #########################################################################################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# gpt = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
# embedding_size = gpt.transformer.wte.weight.shape[1]
# print('embedding_size: ', embedding_size)
# prefix = image embedding, token = text embedding
########################################################################################################
### 官方的Gemma #########################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2b = 2304, 9b = 3584, 27b = 4608
embedding_size = 2304
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
# gemmaConfig = AutoConfig.from_pretrained('google/gemma-2-2b-it')
gemma = Gemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16)

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
# a = count_trainable_parameters(gemma)
# gemma = get_peft_model(gemma, LORAconfig)
# b = count_trainable_parameters(gemma)
# #留下小數點後兩位就好
# percent = round((b / a) * 100, 3)
# print("Before: ", a, "After: ", b, "Percent: ", percent, "%")
########################################################################################################

class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
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
        super(MlpTransformer, self).__init__()
        self.fc1 = nn.Linear(embedding_size, h_dim)
        self.relu = nn.functional.relu
        self.fc2 = nn.Linear(h_dim, embedding_size)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, bias=True, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = embedding_size // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.to_keys_values = nn.Linear(embedding_size, embedding_size * 2, bias=bias)
        self.project = nn.Linear(embedding_size, embedding_size)
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
    def __init__(self,  mlp_ratio=4., bias=False, dropout=0.):
        super(TransformerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_size, eps=eps)
        self.attn = MultiHeadAttention(8, bias=bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(embedding_size, eps=eps)
        self.mlp = MlpTransformer(int(embedding_size * mlp_ratio))

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
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(mlp_ratio))
            elif enc_dec:  # self
                layers.append(TransformerLayer(mlp_ratio))
            else:  # self or cross
                layers.append(TransformerLayer(mlp_ratio))
        self.layers = nn.ModuleList(layers)

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, y, mask)
        return x

class TransformerMapper(nn.Module):
    def __init__(self):
        super(TransformerMapper, self).__init__()
        self.transformer = Transformer(num_layers=8, enc_dec=False)
        self.linear = nn.Linear(512, prefix_length*embedding_size)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, embedding_size), requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], prefix_length, embedding_size)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], -1, -1).to(device).to(torch.bfloat16)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, prefix_length:]
        return out


class Generator(nn.Module):
    def __init__(self, Former, gemma, depth=12):
        super(Generator, self).__init__()
        if Former == "MLP":
            self.Former = MLP()
        else:
            self.Former = TransformerMapper()
        self.gemma = gemma
        self.gemma.eval()
        for param in self.gemma.parameters():
            param.requires_grad = False

        self.Only_image = None
        self.Only_image_text = []
        self.Only_image_loss = []
        self.Empty_Text = None
        self.Empty_Text_text = []
        self.Empty_Text_loss = []
        self.Full_Text = None
        self.Full_Text_text = []
        self.Full_Text_loss = []
        self.generate_beam_output = None
        self.generate_beam_text = []
        self.generate_beam_loss = []
        self.generate2_output = None
        self.generate2_text = []
        self.generate2_loss = []

    def forward(self, image, text_id, mask, mode):
        if mode == 'test':
            dummy = torch.full((image.shape[0], 64), 0, dtype=torch.long).to(device)
            text_embedd = self.gemma.model.embed_tokens(dummy)
        else:
            text_embedd = self.gemma.model.embed_tokens(text_id)
        ##############################################   generate ##############################################
        image_G = self.Former(image)
        embedding_cat = torch.cat((image_G, text_embedd), dim=1)
        ########################################### feature fusion ###########################################
        gemmaOutput = self.gemma(inputs_embeds=embedding_cat, attention_mask=mask)
        logits = gemmaOutput.logits

        return logits


    def generate_woText(self, image, text_id):
        image_GG = self.Former(image)
        gptOutput = self.gemma(inputs_embeds=image_GG)
        logits = gptOutput.logits
        caption_tokens = logits.argmax(-1)[0]
        caption = tokenizer.decode(caption_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        loss = loss_function_10(logits, text_id)
        if self.Only_image != None:
            self.Only_image = torch.cat((self.Only_image, caption_tokens.unsqueeze(0)), dim=0)
        else:
            self.Only_image = caption_tokens.unsqueeze(0)
        self.Only_image_text.append(caption)
        self.Only_image_loss.append(loss.item())
        caption = generate_beam(Generator, tokenizer, embed=image_GG)[0]
        caption_tokens = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=74)['input_ids']
        if self.generate_beam_output != None:
            print(self.generate_beam_output.shape, caption_tokens.shape)
            self.generate_beam_output = torch.cat((self.generate_beam_output, caption_tokens), dim=0)
        else:
            self.generate_beam_output = caption_tokens
        self.generate_beam_text.append(caption)
        caption = generate2(Generator, tokenizer, embed=image_GG)
        caption_tokens = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=74)['input_ids']
        if self.generate2_output != None:
            self.generate2_output = torch.cat((self.generate2_output, caption_tokens), dim=0)
        else:
            self.generate2_output = caption_tokens
        self.generate2_text.append(caption)

    def generate_withText(self, image, text_id, mode, mask=None):
        if mode == 'test':
            dummy = torch.full_like(text_id, 50256, dtype=text_id.dtype).to(device)
            text_embedd = self.gemma.model.embed_tokens(dummy)
        else:
            text_embedd = self.gemma.model.embed_tokens(text_id)
        ##############################################   generate ##############################################
        image_G = self.Former(image)
        ########################################### feature fusion ###########################################
        image_GG = torch.cat((image_G, text_embedd), dim=1)
        gptOutput = self.gemma(inputs_embeds=image_GG, attention_mask=mask)
        logits = gptOutput.logits
        caption_tokens = logits.argmax(-1)[0]
        caption = tokenizer.decode(caption_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(caption)
        print(caption_tokens)
        loss = loss_function(logits, text_id)
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
        def dataframe_Name(name, count=prefix_length+64, rows=1):
            if rows == 1:
                name_df = pd.DataFrame([[name]], columns=["Name"])
                num_df = pd.DataFrame([[0] * count for _ in range(rows)], columns=[f"{i}" for i in range(0, count)])
                name_df = pd.concat([name_df, num_df], axis=1)
                return name_df
            else:
                name_df = pd.DataFrame([[name] * (128 - count) for _ in range(rows)], columns=[f"{i}" for i in range(count, 128)])
                return name_df


        test = pd.DataFrame()
        test['Name'] = ['test1', 'test2','train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7', 'train8', 'train9',
                        'train10']
        Only_image_df = pd.DataFrame(self.Only_image.cpu().detach(), columns=[f"{i}" for i in range(0, prefix_length)])
        Only_image_df = pd.concat([test, Only_image_df, dataframe_Name("-", 64, 12)], axis=1)
        Only_image_df['loss'] = self.Only_image_loss
        Only_image_df['text'] = self.Only_image_text
        generate_beam_df = pd.DataFrame(self.generate_beam_output.cpu().detach(), columns=[f"{i}" for i in range(0, 74)])
        generate_beam_df = pd.concat([test, generate_beam_df], axis=1)
        # generate_beam_df = pd.concat([test, generate_beam_df, dataframe_Name("-", 74, 12)], axis=1)
        generate_beam_df['text'] = self.generate_beam_text
        generate2_df = pd.DataFrame(self.generate2_output.cpu().detach(), columns=[f"{i}" for i in range(0, 74)])
        generate2_df = pd.concat([test, generate2_df], axis=1)
        # generate2_df = pd.concat([test, generate2_df, dataframe_Name("-", 74, 12)], axis=1)
        generate2_df['text'] = self.generate2_text
        print(self.Empty_Text.shape)
        Empty_Text_df = pd.DataFrame(self.Empty_Text.cpu().detach(), columns=[f"{i}" for i in range(0, prefix_length+64)])
        Empty_Text_df = pd.concat([test, Empty_Text_df], axis=1)
        Empty_Text_df['loss'] = self.Empty_Text_loss
        Empty_Text_df['text'] = self.Empty_Text_text
        Full_Text_df = pd.DataFrame(self.Full_Text.cpu().detach(), columns=[f"{i}" for i in range(0, prefix_length+64)])
        Full_Text_df = pd.concat([test, Full_Text_df], axis=1)
        Full_Text_df['loss'] = self.Full_Text_loss
        Full_Text_df['text'] = self.Full_Text_text

        print(dataframe_Name("Only_image").shape, Only_image_df.shape, dataframe_Name("generate_beam").shape,
              generate_beam_df.shape, dataframe_Name("generate2").shape, generate2_df.shape,
              dataframe_Name("Empty_Text").shape, Empty_Text_df.shape, dataframe_Name("Full_Text").shape,
              Full_Text_df.shape)
        final = pd.concat([dataframe_Name("Only_image"), Only_image_df, dataframe_Name("generate_beam"),
                           generate_beam_df, dataframe_Name("generate2"), generate2_df, dataframe_Name("Empty_Text"),
                           Empty_Text_df, dataframe_Name("Full_Text"), Full_Text_df], axis=0)


        print(final.shape)
        print(final.columns)
        final.to_csv('./Model/' + checkpoint_folder + '/' + checkpoint_folder + '_test.csv', index=False)

def generate_beam(
        model,
        tokenizer,
        beam_size: int = 5,
        prompt=None,
        embed=None,
        entry_length=74,
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
            outputs = model.gemma(inputs_embeds=generated)
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
            next_token_embed = model.gemma.model.embed_tokens(next_tokens.squeeze()).view(
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
        entry_length=74,  # maximum number of words
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

                outputs = model.gemma(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gemma.model.embed_tokens(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############# load  model #############
# NetFormer = Former().to(torch.bfloat16).to(device)
Generator = Generator(Former="Trans", gemma=gemma).to(torch.bfloat16).to(device)
#######################################
checkpoint_folder = '20241221_Clip_Clip_Clip_noGemma_prefix10_mask_AdamW'
checkpoint_Generator = './Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetLLM_1.pth'
#######################################
checkpoint_Generator = torch.load(checkpoint_Generator)
Generator.load_state_dict(checkpoint_Generator['model_state_dict'])
#######################################

def loss_function_10(logits, labels):
    logits = logits.contiguous().view(-1, logits.size(-1))
    labels = labels.contiguous().view(-1)
    loss = nn.functional.cross_entropy(logits, labels, ignore_index=0)
    return loss

def loss_function(logits, labels):
    logits = logits[:, prefix_length-1:-1]
    logits = logits.contiguous().view(-1, logits.size(-1))
    labels = labels.contiguous().view(-1)
    loss = nn.functional.cross_entropy(logits, labels, ignore_index=0)
    return loss

# generate
Generator.eval()

print('=======================  test  1 =========================')
image = torch.load('./test_img.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['you all loved it so i brought it back']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)
#
print('=======================  test  2 =========================')
image = torch.load('./test_img2.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['Can’t believe this is real (still) life. The Saucy Nugg legacy is forever cemented in oil on canvas (2024). Thank you']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)



print('=======================  sample1 =========================')
print(train[train['image_id'] == 'bokete_51227'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/imgflip_34.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['You finish doing something at your friends house and look at your phone; 7 missed calls from your mom; 7 missed calls from your mom']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

print('=======================  sample2 =========================')
print(train[train['image_id'] == 'bokete_3820'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/bokete_3820.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['I\'m in my 50s!']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

print('=======================  sample3 =========================')
print(train[train['image_id'] == 'imgflip_0'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/imgflip_0.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['Getting hurt from cutting open your leg; Getting hurt from taking off a bandaid']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

print('=======================  sample4 =========================')
print(train[train['image_id'] == 'imgflip_8'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/imgflip_8.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['image tagged in memes,one does not simply']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

print('=======================  sample5 =========================')
print(train[train['image_id'] == 'imgflip_15'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/imgflip_15.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['KIDS WHEN THEIR PARENTS GIVE THEM; "THE TALK"']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

print('=======================  sample6 =========================')
print(train[train['image_id'] == 'imgflip_19'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/imgflip_19.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['NOT SURE IF PEOPLE ARE UPVOTING MEMES; OR USER NAMES']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

print('=======================  sample7 =========================')
print(train[train['image_id'] == 'bokete_104530'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/bokete_104530.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['It\'s a family night runaway.']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

print('=======================  sample8 =========================')
print(train[train['image_id'] == 'imgflip_730'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/imgflip_730.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['CHUCK IS THE GOOD TYPE OF SCUMBAG; CUZ HE ONLY ROASTS YOU FROM YOUR INSIDES']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

print('=======================  sample9 =========================')
print(train[train['image_id'] == 'imgflip_130'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/imgflip_130.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['SO YOUR TELLIN\' ME THAT SCHOOLS GOOD FOR YOU']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

print('=======================  sample10 =========================')
print(train[train['image_id'] == 'imgflip_677'].shape[0] > 0)
image = torch.load('../../../Oxford_HIC/ImageClip/imgflip_677.pt', weights_only=False).unsqueeze(0).to(device, dtype=torch.bfloat16)
test_gt = ['Y\'ALL GOT ANY MORE OF THEM; JOBS?']
print("ground_truth: ", test_gt)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=prefix_length)
text_id = text_data['input_ids'].to(device)
print("========= Only image =========")
Generator.generate_woText(image, text_id)
text_data = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_data['input_ids'].to(device)
mask = torch.cat((torch.ones(1, prefix_length), text_data['attention_mask']), dim=1).to(device)
print("========= Empty Text =========")
Generator.generate_withText(image, text_id, 'test')
print("========= Full  Text =========")
print(mask.shape)
Generator.generate_withText(image, text_id, 'train', mask)

Generator.print()


