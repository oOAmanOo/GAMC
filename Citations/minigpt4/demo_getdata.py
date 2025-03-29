import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
from skimage import io
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from transformers import StoppingCriteriaList

from Citations.minigpt4.minigpt4.common.config import Config
from Citations.minigpt4.minigpt4.common.dist_utils import get_rank
from Citations.minigpt4.minigpt4.common.registry import registry
from Citations.minigpt4.minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from Citations.minigpt4.minigpt4.datasets.builders import *
from Citations.minigpt4.minigpt4.models import *
from Citations.minigpt4.minigpt4.processors import *
from Citations.minigpt4.minigpt4.runners import *
from Citations.minigpt4.minigpt4.tasks import *

def main():
    # dirPath = '../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
    # dirPath = '../Data/Oxford_HIC/Only10_oxford_hic_data.csv'

    # ff_list=['mcdonalds', 'mcdonalds_switzerland', 'mcdonaldscanada', 'sonicdrivein','wendys']
    insData = 'mcdonalds_switzerland'
    def parse_args():
        parser = argparse.ArgumentParser(description="Demo")
        # parser.add_argument("--cfg-path",  default='./eval_configs/minigpt4_eval.yaml',  help="path to configuration file.")
        parser.add_argument("--cfg-path",  default='./eval_configs/minigpt4_llama2_eval.yaml',  help="path to configuration file.")
        parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )
        args = parser.parse_args()
        return args

    def setup_seeds(config):
        seed = config.run_cfg.seed + get_rank()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cudnn.benchmark = False
        cudnn.deterministic = True


    # ========================================
    #             Model Initialization
    # ========================================

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                 'pretrain_llama2': CONV_VISION_LLama2}

    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    print('Initialization Finished')
    if insData == 'Oxford':
        if os.path.exists(f'../../Data/Oxford_HIC/Minigpt4_{insData}.csv'):
            filtered_data = pd.read_csv(f'../../Data/Oxford_HIC/Minigpt4_{insData}.csv')
            print("loaded")
        else:
            dirPath = f'../../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
            data = pd.read_csv(dirPath)
            data = data.drop(columns=['caption', 'caption_id'])
            print("shape of data: ", data.shape)
            image_id_counts = data['image_id'].value_counts()
            valid_image_ids = image_id_counts[image_id_counts >= 300].index
            print("shape of valid_image_ids: ", valid_image_ids.shape)
            filtered_data = data[data['image_id'].isin(valid_image_ids)]
            print("shape of filtered_data: ", filtered_data.shape)
            filtered_data = filtered_data.drop_duplicates(subset=['image_id'])
            print("shape of filtered_data: ", filtered_data.shape)
            filtered_data['chat'] = ''
            filtered_data['done'] = ''
    else:
        if os.path.exists(f'../../Data/Instagram/Minigpt4_{insData}.csv'):
            filtered_data = pd.read_csv(f'../../Data/Instagram/Minigpt4_{insData}.csv')
        else:
            dirPath = f'../../Data/Instagram/Generate_{insData}.csv'
            data = pd.read_csv(dirPath)
            data = data.drop(columns=['caption'])
            print("shape of data: ", data.shape)
            image_id_counts = data['image_id'].value_counts()
            #####################################################################################################
            valid_image_ids = image_id_counts[image_id_counts >= 100].index
            print("shape of valid_image_ids: ", valid_image_ids.shape)
            filtered_data = data[data['image_id'].isin(valid_image_ids)]
            print("shape of filtered_data: ", filtered_data.shape)
            filtered_data = filtered_data.drop_duplicates(subset=['image_id'])
            print("shape of filtered_data: ", filtered_data.shape)
            filtered_data['chat'] = ''
            filtered_data['done'] = ''

    # ========================================
    #             Gradio Setting
    # ========================================
    num_beams = 1
    temperature = 1.0

    # def gradio_reset(img_list):
    #     if chat_state is not None:
    #         chat_state.messages = []
    #     if img_list is not None:
    #         img_list = []
    #     return chat_state, img_list

    def upload_img(gr_img, chat_state):
        if gr_img is None:
            return chat_state, None
        chat_state = CONV_VISION.copy()
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        chat.encode_img(img_list)
        return chat_state, img_list

    def gradio_ask(user_message, chat_state):
        chat.ask(user_message, chat_state)
        return chat_state

    def gradio_answer(chat_state, img_list, num_beams, temperature):
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=num_beams,
                                  temperature=temperature,
                                  max_new_tokens=300,
                                  max_length=2000)[0]

        return llm_message, chat_state, img_list

        # run without gradio
    filtered_data = filtered_data.reset_index(drop=True)
    # prompt = 'simply categorize the emotions, sentiment and the humor in the image'
    # prompt = 'simply list the emotions, sentiment and the humor in the image'
    prompt1 = 'Analyze the given image and break down its '
    prompt2 = ' aspects by listing their key elements in simple word. Answer None if none exist'

    # prompt = 'Analyze the given image and break down its emotional, sentimental, and humorous aspects by listing their key elements in simple word. Answer None if none exist'

    with tqdm(total=filtered_data.shape[0], leave=True) as progress:
        for i in range(filtered_data.shape[0]):
            llm_message = filtered_data.iloc[i]['chat']
            if filtered_data.iloc[i]['done'] == 'X':
            # if llm_message == '' or pd.isnull(llm_message):
                counter = 0
                llm_message = str(llm_message)
                emotion = str(filtered_data.iloc[i]['emotion'])
                sentiment = str(filtered_data.iloc[i]['sentiment'])
                humor = str(filtered_data.iloc[i]['humor'])
                # print(f'emotion: {emotion}')
                # print(f'sentiment: {sentiment}')
                # print(f'humor: {humor}')
                prompt_mid = ''
                if emotion == 'nan':
                    prompt_mid += 'emotion'
                if sentiment == 'nan':
                    if prompt_mid != '':
                        prompt_mid += ' and '
                    prompt_mid += 'sentiment'
                if humor == 'nan':
                    if prompt_mid != '':
                        prompt_mid += ' and '
                    prompt_mid += 'humor'
                prompt = f"{prompt1}{prompt_mid}{prompt2}"
                # print(prompt)
                while '*' not in llm_message and counter < 5:
                # while ':' not in llm_message and '-' not in llm_message and counter < 3:
                    counter += 1
                    chat_state = None
                    image_id = filtered_data.iloc[i]['image_id']
                    if insData == 'Oxford':
                        filename = f"../../Data/Oxford_HIC/oxford_img/{image_id}.jpg"
                    else:
                        filename = f"../../Data/Instagram/{insData}_img/{image_id}.jpg"
                    image = Image.fromarray(io.imread(filename))
                    chat_state, img_list = upload_img(image, chat_state)
                    # print('=====================================================')
                    # print('Image ID:', image_id)
                    # print('=====================================================')
                    # print('Prompt:', prompt)

                    chat_state = gradio_ask(prompt, chat_state)
                    llm_message, chat_state, img_list = gradio_answer(chat_state, img_list, num_beams, temperature)
                    # print('=====================================================')
                    # print('Chat:', llm_message)
                    # print('=====================================================')
                filtered_data.at[i, 'chat'] = llm_message
                if '*' not in llm_message and counter == 5:
                # if ':' not in llm_message and '-' not in llm_message and counter == 3:
                    filtered_data.at[i, 'done'] = 'X'
                else:
                    filtered_data.at[i, 'done'] = 'done'
                progress.update(1)
            else:
                progress.update(1)
                continue
            if i % 5 == 0:
                if insData == 'Oxford':
                    filtered_data.to_csv(f'../../Data/Oxford_HIC/Minigpt4_{insData}.csv', index=False)
                else:
                    filtered_data.to_csv(f'../../Data/Instagram/Minigpt4_{insData}.csv', index=False)
    if insData == 'Oxford':
        filtered_data.to_csv(f'../../Data/Oxford_HIC/Minigpt4_{insData}.csv', index=False)
    else:
        filtered_data.to_csv(f'../../Data/Instagram/Minigpt4_{insData}.csv', index=False)
if __name__ == '__main__':
    main()