import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms
from transformers import GPT2Tokenizer, GPT2Model
import pandas as pd
import numpy as np
import os
import pickle
import gc
import sys
import argparse
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split


class ImageTextModel(nn.Module):
    def __init__(self, gpt2_model_name="gpt2", feature_dim=512, output_dim=1):
        super(ImageTextModel, self).__init__()

        # Image Encoder (ResNet-50)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, feature_dim)  # Modify FC layer

        # Text Encoder (GPT-2)
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        self.text_proj = nn.Linear(self.gpt2.config.hidden_size, feature_dim)

        # Fusion Layer
        self.fusion = nn.Linear(feature_dim * 2, 128)
        self.classifier = nn.Linear(128, output_dim)

        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification

    def forward(self, image, input_ids, attention_mask):
        # Encode image
        img_features = self.resnet(image)

        # Encode text (Get last hidden state, take CLS token representation)
        text_outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        text_features = self.text_proj(text_features)

        # Fusion (Concatenation + Projection)
        fused = torch.cat((img_features, text_features), dim=1)
        fused = self.fusion(fused)
        fused = torch.relu(fused)

        # Classification
        logits = self.classifier(fused)
        probs = self.sigmoid(logits)

        return probs

class Dataset(torch.utils.data.Dataset):
    def get_image_features(self, img_id, humor):
        if humor == 0:
            file = f"../Data/humorscore_image_{self.traintest}_data/COCO_{int(img_id):012d}.pt"
            if os.path.exists(file):
                return torch.load(file)
            else:
                filename = f"{self.coco_image_dir}{int(img_id):012d}.jpg"
                image = Image.open(filename).convert('RGB')
                image = self.image_transform(image)
                torch.save(image, f'../Data/humorscore_image_{self.traintest}_data/COCO_{int(img_id):012d}.pt')
            return image
        else:
            file = f"../Data/humorscore_image_{self.traintest}_data/oxford_{img_id}.pt"
            if os.path.exists(file):
                return torch.load(file)
            else:
                filename = f"../Data/Oxford_HIC/oxford_img/{img_id}.jpg"
                image = Image.open(filename).convert('RGB')
                image = self.image_transform(image)
                torch.save(image, f'../Data/humorscore_image_{self.traintest}_data/oxford_{img_id}.pt')
                return image

    def get_caption_embedding(self, caption):
        inputs = self.tokenizer([caption], truncation=True, max_length=64, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        padding = torch.zeros(1, 64 - input_ids.shape[1], dtype=torch.int64)
        input_ids = torch.cat((input_ids, padding), dim=1)
        attention_mask = torch.cat((attention_mask, padding), dim=1)
        return input_ids.squeeze(0), attention_mask.squeeze(0)

    def __getitem__(self, item: int):
        image = self.get_image_features(self.image_list[item], self.humor[item])
        caption_id, caption_attmask = self.get_caption_embedding(self.caption_list[item])
        humor = self.humor[item]
        return image, caption_id, caption_attmask, humor

    def __len__(self):
        return len(self.image_list)

    def __init__(self, oxford_data: pd.DataFrame, traintest: str, dataPath: str):
        device = torch.device('cuda:0')
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 doesn’t have a pad token by default
        self.traintest = traintest
        # Define image transform
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_list = []
        self.caption_list = []
        self.humor = []
        if traintest == 'train':
            out_path = f"../Data/{dataPath}_train.pkl"
            self.coco_dir = 'C:/Users/user/fiftyone/coco-2014/raw/captions_train2014.json'
            self.coco_image_dir = 'C:/Users/user/fiftyone/coco-2014/train/data/COCO_train2014_'
        else:
            out_path = f"../Data/{dataPath}_test.pkl"
            self.coco_dir = 'C:/Users/user/fiftyone/coco-2014/raw/captions_val2014.json'
            self.coco_image_dir = 'C:/Users/user/fiftyone/coco-2014/validation/data/COCO_val2014_'

        if os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                alldata = pickle.load(f)
            self.image_list = alldata['image_list']
            self.caption_list = alldata['caption_list']
            self.humor = alldata['humor']
            print('Data Loaded')
            print("%0d embeddings saved " % len(self.image_list))
        else:
            with open(self.coco_dir, 'r') as f:
                data = json.load(f)
            data = data['annotations']
            print("%0d captions loaded from json " % len(data))

            with tqdm(total=len(data)) as pbar:
                for i in range(len(data)):
                    if traintest == 'test' and i > 414113/4:
                        break
                    d = data[i]
                    self.image_list.append(d["image_id"])
                    self.caption_list.append(d['caption'])
                    self.humor.append(torch.tensor([0]))
                    if (i + 1) % 10000 == 0:
                        with open(out_path, 'wb') as f:
                            pickle.dump({"image_list": self.image_list,
                                         "caption_list": self.caption_list,
                                         "humor": torch.cat(self.humor, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
                        pbar.set_postfix({"present": i})
                    pbar.update(1)
            with open(out_path, 'wb') as f:
                pickle.dump({"image_list": self.image_list,
                             "caption_list": self.caption_list,
                             "humor": torch.cat(self.humor, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
            print('COCO Done')
            print("%0d embeddings saved " % len(self.image_list))
            pbar.close()

            with tqdm(total=len(oxford_data)) as pbar:
                for i in range(len(oxford_data)):
                    d = oxford_data.iloc[i]
                    d = d.to_dict()
                    self.image_list.append(d["image_id"])
                    self.caption_list.append(d['caption'])
                    self.humor.append(torch.tensor([1]))
                    if (i + 1) % 10000 == 0:
                        with open(out_path, 'wb') as f:
                            pickle.dump({"image_list": self.image_list,
                                         "caption_list": self.caption_list,
                                         "humor": torch.cat(self.humor, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
                        pbar.set_postfix({"present": i})
                    pbar.update(1)
            with open(out_path, 'wb') as f:
                pickle.dump({"image_list": self.image_list,
                             "caption_list": self.caption_list,
                             "humor": torch.cat(self.humor, dim=0)}, f, pickle.HIGHEST_PROTOCOL)
            print('Oxford Done')
            print("%0d embeddings saved " % len(self.image_list))
            pbar.close()

def train(model, args, output_dir: str = ".", output_prefix: str = ""):
    train_losses = []
    test_losses = []
    best_train_loss = 9999999999
    best_test_loss = 9999999999
    save = []

    device = torch.device('cuda:0')
    batch_size = args.bs

    model = model.to(device)

    dataPath = args.dataPath
    if os.path.exists(f"../Data/{dataPath}_train.pkl") and os.path.exists(f"../Data/{dataPath}_test.pkl"):
        if os.path.exists(f"../Data/{dataPath}_train.pkl"):
            trainDataset = Dataset(pd.DataFrame(), 'train', dataPath)
        if os.path.exists(f"../Data/{dataPath}_test.pkl"):
            testDataset = Dataset(pd.DataFrame(), 'test', dataPath)
    else:
        dirPath = '../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
        data = pd.read_csv(dirPath)
        threshold = data['funny_score'].quantile(0.75)
        data = data[data['funny_score'] >= threshold]
        unique_image_ids = data['image_id'].unique()
        train_ids, test_ids = train_test_split(unique_image_ids, test_size=0.2, random_state=42)
        train = data[data['image_id'].isin(train_ids)]
        test = data[data['image_id'].isin(test_ids)]
        print(train.shape, test.shape)
        trainDataset = Dataset(train, 'train', dataPath)
        testDataset = Dataset(test, 'test', dataPath)
    # get data size
    print(len(trainDataset), len(testDataset))
    train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, drop_last=True)
    epoch = 0
    while len(trainDataset) > batch_size and len(testDataset) > batch_size:
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.BCELoss()
        print(f">>> Training epoch {epoch + 1}")
        sys.stdout.flush()
        trainLoss = 0
        testLoss = 0
        model.train()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (images, caption_ids, caption_masks, humor) in enumerate(train_dataloader):
            model.zero_grad()
            images, caption_ids, caption_masks = images.to(device), caption_ids.to(device), caption_masks.to(device)
            humor = humor.unsqueeze(1).to(device, dtype=torch.float)
            outputs = model(images, caption_ids, caption_masks)
            loss = criterion(outputs, humor)
            trainLoss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            # if idx % 101 == 100:
            #     break
        trainLoss /= len(train_dataloader)
        train_losses.append(trainLoss)
        progress.set_postfix({"loss": trainLoss})
        progress.close()

        model.eval()
        with torch.no_grad():
            progress = tqdm(total=len(test_dataloader), desc=output_prefix)
            for idx, (images, caption_ids, caption_masks, humor) in enumerate(test_dataloader):
                model.zero_grad()
                images, caption_ids, caption_masks = images.to(device), caption_ids.to(device), caption_masks.to(device)
                humor = humor.unsqueeze(1).to(device, dtype=torch.float)
                outputs = model(images, caption_ids, caption_masks)
                loss = criterion(outputs, humor)
                testLoss += loss.item()
                progress.set_postfix({"loss": loss.item()})
                progress.update()
        testLoss /= len(test_dataloader)
        test_losses.append(testLoss)
        progress.set_postfix({"loss": testLoss})
        progress.close()

        if trainLoss < best_train_loss and testLoss < best_test_loss:
            best_train_loss = trainLoss
            best_test_loss = testLoss
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch + 1:03d}.pt"),
            )
            save.append('V')
        else:
            save.append(' ')

        loss_data = pd.DataFrame()
        loss_data['train_loss'] = train_losses
        loss_data['test_loss'] = test_losses
        loss_data['save'] = save
        loss_data.to_csv(f"{output_dir}/{output_prefix}-loss.csv", index=False)

        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.legend()
        plt.savefig(f"{output_dir}/{output_prefix}-loss.png")
        plt.show()

        epoch += 1
    return model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default='humorScore_oxford_with_coco', help='data path')
    parser.add_argument('--out_dir', default='humorScore_20250310_oxford_with_coco', help='output directory')
    parser.add_argument('--prefix', default='checkpoint', help='prefix for saved filenames')
    parser.add_argument('--bs', type=int, default=128)
    args = parser.parse_args()
    if not os.path.exists('./Model/' + args.out_dir):
        os.makedirs('./Model/' + args.out_dir)
        os.makedirs('D:/MemeGAN/Model/' + args.out_dir)
    args.out_dir = './Model/' + args.out_dir

    model = ImageTextModel()
    device = torch.device('cuda:0')
    model = model.to(device)
    # 20250115_totalClip_oxford_only100_300k_transformer_p40_falcon_bleu1_0.05 == 32
    # save_file = '20250204_totalClip_oxford_1000up_only1000_rest_300up_top300_transformer_p64_falcon_swin_tf8_ins'
    # i = 27
    # model.load_state_dict(torch.load(f'./Model/{save_file}/checkpoint-{i:03d}.pt'))
    model.eval()
    train(model, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()