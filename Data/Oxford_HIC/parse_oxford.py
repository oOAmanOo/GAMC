import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    # with open('C:/Users/user/fiftyone/coco-2014/raw/captions_train2014.json', 'r') as f:
    #     data = json.load(f)
    # data = data['annotations']
    dirPath = './CaptionID_oxford_hic_data.csv'
    data = pd.read_csv(dirPath)
    image_id_counts = data['image_id'].value_counts()
    data['caption'] = data['caption'].str.lower()

    valid_image_ids = image_id_counts[image_id_counts >= 800].index
    print("shape of valid_image_ids: ", valid_image_ids.shape)
    filtered_data = data[data['image_id'].isin(valid_image_ids)]
    print("shape of filtered_data: ", data.shape)
    train = (
        filtered_data.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
        .groupby('image_id')
        .head(800)
    )
    print("shape of train: ", train.shape)
    #
    valid_image_ids = image_id_counts[(image_id_counts >= 300) & (image_id_counts < 800)].index
    print("shape of valid_image_ids: ", valid_image_ids.shape)
    filtered_data = data[data['image_id'].isin(valid_image_ids)]
    humorscore_avg = filtered_data.groupby('image_id')['funny_score'].mean()
    top_humor_image_image_id = humorscore_avg.nlargest(56).index
    print("len of top_humor_image_image_id: ", len(top_humor_image_image_id))
    filtered_data = filtered_data[filtered_data['image_id'].isin(top_humor_image_image_id)]
    print("shape of filtered_data: ", filtered_data.shape)
    test = (
        filtered_data.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
        .groupby('image_id')
        .head(300)
    )
    print("shape of test: ", test.shape)
    ######################################################################################################
    # train = pd.DataFrame()
    # test = pd.DataFrame()
    # for image_id, group in data.groupby("image_id"):
    #     train_split, test_split = train_test_split(group, test_size=0.2, random_state=42)
    #     train = pd.concat([train, train_split])
    #     test = pd.concat([test, test_split])
    # print(f'train: {train.shape}')
    # print(f'test: {test.shape}')
    ######################################################################################################
    # data = data.sample(n=300000, random_state=42, replace=True).reset_index(drop=True)
    # print("sample of data: ", data.shape)
    # train, test = train_test_split(data, test_size=0.2, random_state=42)
    # print(data.shape)
    # print("%0d captions loaded from json " % len(data))
    ######################################################################################################
    # unique_image_ids = data['image_id'].unique()
    # # unique_image_ids = unique_image_ids[:30000]
    # # unique_image_ids, rest = train_test_split(unique_image_ids, test_size=0.7, random_state=42)
    # # print(unique_image_ids.shape)
    # train_ids, test_ids = train_test_split(unique_image_ids, test_size=0.2, random_state=42)
    # train = data[data['image_id'].isin(train_ids)]
    # test = data[data['image_id'].isin(test_ids)]
    # print(train.shape, test.shape)
    ######################################################################################################

    def parse(out_path, data):
        remove_counter = 0
        all_embeddings = []
        all_captions = []
        all_funnyscore = []
        for i in tqdm(range(len(data))):
            d = data.iloc[i]
            d = d.to_dict()
            img_id = d["image_id"]
            funnyscore = d["funny_score"]
            filename = f"./oxford_img/{img_id}.jpg"
            image = io.imread(filename)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            d["clip_embedding"] = i
            all_embeddings.append(prefix)
            all_captions.append(d)
            all_funnyscore.append(torch.tensor([funnyscore]).unsqueeze(0))
            if (i + 1) % 10000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, "funnyscore": torch.cat(all_funnyscore, dim=0)}, f)
        with open(out_path, 'wb') as f:
            pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, "funnyscore": torch.cat(all_funnyscore, dim=0)}, f)
        print('Done')
        print("%0d embeddings saved| " % len(all_embeddings))
        return remove_counter
    out_path_train = f"./parse/oxford_800_8_300_2_{clip_model_name}_train.pkl"
    out_path_test = f"./parse/oxford_800_8_300_2_{clip_model_name}_test.pkl"
    print(f'train remove captions: {parse(out_path_train, train)}')
    print(f'test remove captions: {parse(out_path_test, test)}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
