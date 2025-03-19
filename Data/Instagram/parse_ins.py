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

    dirPath = './Generate_sonicdrivein.csv'
    data = pd.read_csv(dirPath)
    print("shape of data: ", data.shape)
    image_id_counts = data['image_id'].value_counts()
    threshold = 24

    original = pd.read_csv('./Filter_sonicdrivein.csv')
    original['caption'] = original['caption'].str.lower()
    data['caption'] = data['caption'].str.lower()
    original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
    original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)
    original = original[original['gen_count'] >= 100]
    print('shape of 100up: ', original.shape, 'images:', len(original['image_id'].unique()))
    print('funny:', original[original['funny_score'] == 1].shape[0], 'else:', original[original['funny_score'] != 1].shape[0])
    original_train = original[original['text_len'] >= threshold]
    print('train_funny:' , original_train[original_train['funny_score'] == 1].shape[0], 'train_else:', original_train[original_train['funny_score'] != 1].shape[0])
    train = data.merge(original_train, on='image_id', how='inner', suffixes=('', '_'))
    print("shape of train: ", train.shape, 'images:', len(train['image_id'].unique()))
    train = (
        train.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
        .groupby('image_id')
        .head(200)
    )
    print("shape of train: ", train.shape, 'images:', len(train['image_id'].unique()))
    original_test = original[original['text_len'] < threshold]
    print("shape of test: ", original_test.shape, 'images:', len(original_test['image_id'].unique()))
    print('funny:', original_test[original_test['funny_score'] == 1].shape[0], 'else:', original_test[original_test['funny_score'] != 1].shape[0])
    # get  top 200 datas on funnyscore
    test = original_test.sort_values(by=['funny_score'], ascending=[False])[:(len(train['image_id'].unique())//4)]
    print("shape of test: ", test.shape, 'images:', len(test['image_id'].unique()))
    print('funny:', test[test['funny_score'] == 1].shape[0], 'else:', test[test['funny_score'] != 1].shape[0])

    # ######################################################################################################
    # valid_image_ids = image_id_counts[image_id_counts >= 100].index
    # print("shape of valid_image_ids: ", valid_image_ids.shape)
    # filtered_data = data[data['image_id'].isin(valid_image_ids)]
    # print("shape of filtered_data: ", filtered_data.shape)
    # data = (
    #     filtered_data.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
    #     .groupby('image_id')
    #     .head(200)
    # )
    # print("shape of train: ", data.shape)

    # valid_image_ids = image_id_counts[(image_id_counts >= 50) & (image_id_counts < 100)].index
    # print("shape of valid_image_ids: ", valid_image_ids.shape)
    # filtered_data = data[data['image_id'].isin(valid_image_ids)]
    # print("shape of filtered_data: ", filtered_data.shape)
    # test = (
    #     filtered_data.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
    #     .groupby('image_id')
    #     .head(50)
    # )
    # print("shape of test: ", test.shape)
    ######################################################################################################
    # print("=============== Train ================")
    # image_id_counts = data['image_id'].value_counts()
    # print(f'Number of unique image_id: {len(image_id_counts)}')
    # valid_image_ids = image_id_counts[image_id_counts >= 300].index
    # print(f'Number of image_id with 300 captions: {len(valid_image_ids)}')
    # filtered_data = data[data['image_id'].isin(valid_image_ids)]
    # print(f'Number of data all: {filtered_data.shape[0]}')
    # train = (
    #     filtered_data.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
    #     .groupby('image_id')
    #     .head(300)
    # )
    # print(f'Number of data 300: {train.shape[0]}')
    #
    # print("=============== Test ================")
    # valid_image_ids = image_id_counts[(image_id_counts >= 200) & (image_id_counts < 300)].index
    # print(f'Number of image_id with 200 captions: {len(valid_image_ids)}')
    # # 篩選原始資料
    # filtered_data = data[data['image_id'].isin(valid_image_ids)]
    # print(f'Number of data all: {filtered_data.shape[0]}')
    # test = (
    #     filtered_data.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
    #     .groupby('image_id')
    #     .head(200)
    # )
    # print(f'Number of data 200: {test.shape[0]}')
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
    # train_ids, test_ids = train_test_split(unique_image_ids, test_size=0.2, random_state=42)
    # train = data[data['image_id'].isin(train_ids)]
    # image_id_counts = train['image_id'].value_counts()
    # print(f'train image_id_counts: ', len(image_id_counts))
    # test = data[data['image_id'].isin(test_ids)]
    # image_id_counts = test['image_id'].value_counts()
    # print(f'test image_id_counts: ', len(image_id_counts))
    # print(train.shape, test.shape)
    ######################################################################################################

    def parse(out_path, data):
        all_embeddings = []
        all_captions = []
        all_funnyscore = []
        for i in tqdm(range(len(data))):
            d = data.iloc[i]
            d = d.to_dict()
            img_id = d["image_id"]
            funnyscore = d["funny_score"]
            filename = f"./sonicdrivein_img/{img_id}.jpg"
            image = io.imread(filename)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            d["clip_embedding"] = i
            all_embeddings.append(prefix)
            all_captions.append(d)
            all_funnyscore.append(funnyscore)
            if (i + 1) % 10000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, "funnyscore": all_funnyscore}, f)
        with open(out_path, 'wb') as f:
            pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, "funnyscore": all_funnyscore}, f)
        print('Done')
        print("%0d embeddings saved " % len(all_embeddings))
        return 0
    out_path_train = f"./parse/100up_passlength_o_{threshold}_sonicdrivein_{clip_model_name}_train.pkl"
    out_path_test = f"./parse/100up_passlength_x_{threshold}_sonicdrivein_{clip_model_name}_test.pkl"
    parse(out_path_train, train)
    parse(out_path_test, test)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
