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
    # Generate >> augmented data
    dirPath = './Generate_mcdonalds_switzerland.csv'
    data = pd.read_csv(dirPath)
    data['caption'] = data['caption'].str.lower()
    print("shape of data: ", data.shape)
    image_id_counts = data['image_id'].value_counts()
    # Filter >> original data remove duplicates image caption pairs
    original = pd.read_csv('./Filter_mcdonalds_switzerland.csv')
    original['caption'] = original['caption'].str.lower()
    original['text_len'] = original['caption'].apply(lambda x: len(x.split()))
    original['gen_count'] = original['image_id'].apply(lambda x: image_id_counts[x] if x in image_id_counts else 0)

    """
        Filter data with following conditions:
            McDonalds Switzerland: 53 funny image caption pairs, 171 other image caption pairs
                -- gen_count >= 100
                -- text_len >= 12
            Sonic Drive-In: 169 funny image caption pairs, 55 other image caption pairs
                -- gen_count >= 100
                -- text_len >= 10
    """
    original = original[original['gen_count'] >= 100]
    print('shape of 100up: ', original.shape, 'images:', len(original['image_id'].unique()))
    threshold = 12
    original_train = original[original['text_len'] >= threshold]
    original_train_funny = original_train[original_train['funny_score'] == 1]
    original_train_others = original_train[original_train['funny_score'] != 1]
    print('funny:', len(original_train_funny['image_id'].unique()), 'else:', len(original_train_others['image_id'].unique()))
    original_train = pd.concat([original_train_funny, original_train_others[:171]])
    print("shape of original_train: ", original_train.shape, 'images:', len(original_train['image_id'].unique()))
    print('train_funny:' , original_train[original_train['funny_score'] == 1].shape[0], 'train_else:', original_train[original_train['funny_score'] != 1].shape[0])
    train = data.merge(original_train, on='image_id', how='inner', suffixes=('', '_'))
    print("shape of train: ", train.shape, 'images:', len(train['image_id'].unique()))
    # get 200 datas ( original data +0, augmented data -0.1)
    train = (
        train.sort_values(by=['image_id', 'funny_score'], ascending=[True, False])
        .groupby('image_id')
        .head(200)
    )
    print("shape of train: ", train.shape, 'images:', len(train['image_id'].unique()))
    original_test = original[original['text_len'] < threshold]
    print("shape of test: ", original_test.shape, 'images:', len(original_test['image_id'].unique()))
    print('funny:', original_test[original_test['funny_score'] == 1].shape[0], 'else:', original_test[original_test['funny_score'] != 1].shape[0])
    # 80 20 split, test is 1/4 of train >> test
    # all data >> testAll
    test = original_test.sort_values(by=['funny_score'], ascending=[False])[:(len(train['image_id'].unique())//4)]
    print("shape of test: ", test.shape, 'images:', len(test['image_id'].unique()))
    print('funny:', test[test['funny_score'] == 1].shape[0], 'else:', test[test['funny_score'] != 1].shape[0])


    def parse(out_path, data):
        all_embeddings = []
        all_captions = []
        all_funnyscore = []
        for i in tqdm(range(len(data))):
            d = data.iloc[i]
            d = d.to_dict()
            img_id = d["image_id"]
            funnyscore = d["funny_score"]
            filename = f"./mcdonalds_switzerland_img/{img_id}.jpg"
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
        print("%0d embeddings saved " % len(all_embeddings))
        return 0

    out_path_train = f"./parse/100up_only200_lessNotFunImg_53_171_passlength_{threshold}_o_mcdonalds_switzerland_{clip_model_name}_train.pkl"
    out_path_test = f"./parse/100up_only200_lessNotFunImg_53_171_passlength_{threshold}_x_mcdonalds_switzerland_{clip_model_name}_test.pkl"
    # out_path_test = f"./parse/100up_only200_lessNotFunImg_53_171_passlength_{threshold}_x_mcdonalds_switzerland_{clip_model_name}_testAll.pkl"
    parse(out_path_train, train)
    parse(out_path_test, test)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
