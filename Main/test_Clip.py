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
    def __init__(self, Former, gpt, depth=12):
        super(Generator, self).__init__()
        self.Former = Former
        self.gpt = gpt
        self.gpt.eval()
        self.gptLinearBefore = nn.Linear(768, gpt_embedding_size)
        # feed forward
        self.feedForwardLinear = nn.Linear(768, 768)
        self.feedForwardLayerNorm = nn.LayerNorm(768, eps=eps)

    def forward(self, image, text_id=None):
        if text_id is None:
            text_id = torch.zeros((image.shape[0], 64), dtype=torch.long).to(device)
        text_embedd = gpt.transformer.wte(text_id)
        ##############################################   generate ##############################################
        image = image.transpose(0, 1)
        image_G = self.Former.image(image)
        ########################################### feature fusion ###########################################
        text_embedd = text_embedd.transpose(0, 1)
        text_G = self.Former.text(text_embedd)
        image_G = image_G.transpose(0, 1)
        text_G = text_G.transpose(0, 1)
        feature_fusion = image_G + text_G  # visual_attending_textual + textual_attending_visual
        feature_fusionFF = self.feedForwardLinear(feature_fusion)
        feature_fusion_final = self.feedForwardLayerNorm(feature_fusion + feature_fusionFF)
        image_GG = self.gptLinearBefore(feature_fusion_final)
        gotOutput = self.gpt(inputs_embeds=image_GG, labels=text_id)
        logits = gotOutput.logits
        return logits
    def generate_woText(self, image):
        image = image.transpose(0, 1)
        image_G = self.Former.image(image)
        image_G = image_G.transpose(0, 1)
        image_GG = self.gptLinearBefore(image_G)
        print("generate_beam", generate_beam(Generator, tokenizer, embed=image_GG)[0])
        print("generate2", generate2(Generator, tokenizer, embed=image_GG))

    def generate_withText(self, image, text_id=None):
        if text_id is None:
            text_id = torch.zeros((image.shape[0], 64), dtype=torch.long).to(device)
        text_embedd = gpt.transformer.wte(text_id)
        ##############################################   generate ##############################################
        image = image.transpose(0, 1)
        image_G = self.Former.image(image)
        ########################################### feature fusion ###########################################
        text_embedd = text_embedd.transpose(0, 1)
        text_G = self.Former.text(text_embedd)
        image_G = image_G.transpose(0, 1)
        text_G = text_G.transpose(0, 1)
        feature_fusion = image_G + text_G  # visual_attending_textual + textual_attending_visual
        feature_fusionFF = self.feedForwardLinear(feature_fusion)
        feature_fusion_final = self.feedForwardLayerNorm(feature_fusion + feature_fusionFF)
        image_GG = self.gptLinearBefore(feature_fusion_final)
        print("generate_beam", generate_beam(Generator, tokenizer, embed=image_GG)[0])
        print("generate2", generate2(Generator, tokenizer, embed=image_GG))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############# load  model #############
NetFormer = Former().to(torch.bfloat16).to(device)
Generator = Generator(Former=NetFormer, gpt=gpt).to(torch.bfloat16).to(device)
#######################################
checkpoint_folder = '20241211_LoRa_id_shortPrompt_CE_base_20241201_wo_coAttention_temp'
checkpoint_Generator = './Model/' + checkpoint_folder + '/' + checkpoint_folder + '_NetLLM_1.pth'
#######################################
checkpoint_Generator = torch.load(checkpoint_Generator)
Generator.load_state_dict(checkpoint_Generator['model_state_dict'])
#######################################

def loss_function(logits, text_id):
    logit = logits.view(-1, logits.size(-1))
    text_id = text_id.view(-1)
    text_id[text_id == 50256] = -100
    loss = nn.CrossEntropyLoss()(logit, text_id)
    return loss

# generate
Generator.eval()

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
                    nn.softmax(sorted_logits, dim=-1), dim=-1
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

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


print('=======================  test  1 =========================')
image = imageExtraction("./test_img.jpg").to(torch.bfloat16)
test_gt = ['you all loved it so i brought it back']
print("ground_truth: ", test_gt)
text_id = tokenizer(test_gt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
text_id = text_id['input_ids'].to(device)
print("=== Only image ===")
Generator.generate_woText(image)
print("=== Empty Text ===")
Generator.generate_withText(image)
print("=== Full  Text ===")
Generator.generate_withText(image, text_id)



# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# #
# image = imageExtraction("./test_img2.jpg").expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['Can’t believe this is real (still) life. The Saucy Nugg legacy is forever cemented in oil on canvas (2024). Thank you']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())



# print('=======================  sample1 =========================')
# print(train[train['image_id'] == 'bokete_51227'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/bokete_51227.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['Matsui found a porn actress in the audience.']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# print('=======================  sample2 =========================')
# print(train[train['image_id'] == 'bokete_3820'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/bokete_3820.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['I\'m in my 50s!']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# print('=======================  sample3 =========================')
# print(train[train['image_id'] == 'imgflip_0'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/imgflip_0.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['Getting hurt from cutting open your leg; Getting hurt from taking off a bandaid']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# print('=======================  sample4 =========================')
# print(train[train['image_id'] == 'imgflip_8'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/imgflip_8.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['image tagged in memes,one does not simply']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# print('=======================  sample5 =========================')
# print(train[train['image_id'] == 'imgflip_15'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/imgflip_15.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['KIDS WHEN THEIR PARENTS GIVE THEM; "THE TALK"']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# print('=======================  sample6 =========================')
# print(train[train['image_id'] == 'imgflip_19'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/imgflip_19.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['NOT SURE IF PEOPLE ARE UPVOTING MEMES; OR USER NAMES']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# print('=======================  sample7 =========================')
# print(train[train['image_id'] == 'bokete_104530'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/bokete_104530.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['It\'s a family night runaway.']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# print('=======================  sample8 =========================')
# print(train[train['image_id'] == 'imgflip_730'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/imgflip_730.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['CHUCK IS THE GOOD TYPE OF SCUMBAG; CUZ HE ONLY ROASTS YOU FROM YOUR INSIDES']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# print('=======================  sample9 =========================')
# print(train[train['image_id'] == 'imgflip_130'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/imgflip_130.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['SO YOUR TELLIN\' ME THAT SCHOOLS GOOD FOR YOU']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())
# print('=======================  sample10 =========================')
# print(train[train['image_id'] == 'imgflip_677'].shape[0] > 0)
# image = torch.load('../../Oxford_HIC/ImageData/imgflip_677.pt', weights_only=False).unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)
# test_gt = ['Y\'ALL GOT ANY MORE OF THEM; JOBS?']
# text_id = textExtraction(tokenizer, gemmaConfig, test_gt).expand(batch_size, -1)
# gemma_loss, logits = Generator(image.to(device), text_id.to(device), prompt_gemma.detach())
# loss = loss_function(logits, text_id.to(device))
# loss_next = loss_function_next(logits, text_id.to(device))
# print("gemma_loss", gemma_loss.item(), "loss", loss.item(), "loss_next", loss_next.item())

