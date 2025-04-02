import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
from skimage import io
from PIL import Image
import re
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
    insData = 'sentence_generate_Oxford'
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
    if 'Oxford' in insData :
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

    def addwhat(text):
        text = text.replace('*', '').replace('\n', '').strip()
        remove = re.findall(r'\d.', text)
        for i in remove:
            text = text.replace(i, '').strip()
        return text

    def categorize(text, emotion, sentiment, humor):
        if text == 'nan':
            return emotion, sentiment, humor
        text = text.lower()
        split1 = text.split("\n\n")
        if len(split1) == 1:
            split1 = text.split("\r\n\r\n")
        category = ''
        emotion_skip = False
        sentiment_skip = False
        humor_skip = False
        if emotion == '':
            emotion = ''
        else:
            emotion_skip = True
        if sentiment == '':
            sentiment = ''
        else:
            sentiment_skip = True
        if humor == '':
            humor = ''
        else:
            humor_skip = True
        for i in range(len(split1)):
            check_emotion = re.findall(r'emotion\b', split1[i])
            if len(check_emotion) == 0:
                check_emotion = re.findall(r'emotions\b', split1[i])
            check_sentiment = re.findall(r'sentiment\b', split1[i])
            if len(check_sentiment) == 0:
                check_sentiment = re.findall(r'sentiments\b', split1[i])
            check_humor = re.findall(r'humor\b', split1[i])

            if category == '':
                # if 'emotion' in split1[i] and 'sentiment' not in split1[i] and 'humor' not in split1[i]:
                if len(check_emotion) > 0 and len(check_sentiment) == 0 and len(check_humor) == 0:
                    if split1[i][-1] == ':':
                        category = 'emotion'
                    elif emotion_skip == False:
                        if 'emotion:' in split1[i]:
                            split2 = split1[i].split('emotion:')
                            emotion += addwhat(split2[1])
                        elif 'emotions:' in split1[i]:
                            split2 = split1[i].split('emotions:')
                            emotion += addwhat(split2[1])
                        else:
                            emotion += addwhat(split1[i])
                    continue
                # elif 'emotion' not in split1[i] and 'sentiment' in split1[i] and 'humor' not in split1[i]:
                elif len(check_emotion) == 0 and len(check_sentiment) != 0 and len(check_humor) == 0:
                    if split1[i][-1] == ':':
                        category = 'sentiment'
                    elif sentiment_skip == False:
                        if 'sentiment:' in split1[i]:
                            split2 = split1[i].split('sentiment:')
                            sentiment += addwhat(split2[1])
                        else:
                            sentiment += addwhat(split1[i])
                    continue
                # elif 'emotion' not in split1[i] and 'sentiment' not in split1[i] and 'humor' in split1[i]:
                elif len(check_emotion) == 0 and len(check_sentiment) == 0 and len(check_humor) != 0:
                    if split1[i][-1] == ':':
                        category = 'humor'
                    elif humor_skip == False:
                        if 'humor:' in split1[i]:
                            split2 = split1[i].split('humor:')
                            humor += addwhat(split2[1])
                        else:
                            humor += addwhat(split1[i])
                    continue
                elif len(check_emotion) == 0 and len(check_sentiment) == 0 and len(check_humor) == 0:
                    check_humor = re.findall(r'humorous\b', split1[i])
                    if len(check_emotion) == 0 and len(check_sentiment) == 0 and len(check_humor) != 0:
                        if split1[i][-1] == ':':
                            category = 'humor'
                        elif humor_skip == False:
                            if 'humor:' in split1[i]:
                                split2 = split1[i].split('humor:')
                                humor += addwhat(split2[1])
                            else:
                                humor += addwhat(split1[i])
                        continue
            else:
                if category == 'emotion' and emotion_skip == False:
                    emotion += addwhat(split1[i])
                elif category == 'sentiment' and sentiment_skip == False:
                    sentiment += addwhat(split1[i])
                elif category == 'humor' and humor_skip == False:
                    humor += addwhat(split1[i])
                category = ''
        return emotion, sentiment, humor

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
                emotion = ('' if str(filtered_data.iloc[i]['emotion']) == 'nan' else str(filtered_data.iloc[i]['emotion']))
                sentiment = ('' if str(filtered_data.iloc[i]['sentiment']) == 'nan' else str(filtered_data.iloc[i]['sentiment']))
                humor = ('' if str(filtered_data.iloc[i]['humor']) == 'nan' else str(filtered_data.iloc[i]['humor']))
                # print(f'emotion: {emotion}')
                # print(f'sentiment: {sentiment}')
                # print(f'humor: {humor}')
                prompt_mid = ''
                if emotion == '':
                    prompt_mid += 'emotion'
                if sentiment == '':
                    if prompt_mid != '':
                        prompt_mid += ' and '
                    prompt_mid += 'sentiment'
                if humor == '':
                    if prompt_mid != '':
                        prompt_mid += ' and '
                    prompt_mid += 'humor'
                prompt = f"{prompt1}{prompt_mid}{prompt2}"
                # print(prompt)

                chat_state = None
                image_id = filtered_data.iloc[i]['image_id']

                if 'Oxford' in insData:
                    filename = f"../../Data/Oxford_HIC/oxford_img/{image_id}.jpg"
                else:
                    filename = f"../../Data/Instagram/{insData}_img/{image_id}.jpg"
                image = Image.fromarray(io.imread(filename)).convert("RGB")
                chat_state, img_list = upload_img(image, chat_state)
                # print('=====================================================')
                # print('Image ID:', image_id)
                # print('=====================================================')
                # print('Prompt:', prompt)
                # while '*' not in llm_message and counter < 3:
                # while ':' not in llm_message and '-' not in llm_message and counter < 3:
                # print(len(emotion) == 0 ,len(sentiment) == 0,len(humor) == 0)
                while (len(emotion) == 0 or len(sentiment) == 0 or len(humor) == 0) and counter < 5:
                    counter += 1
                    chat_state = gradio_ask(prompt, chat_state)
                    llm_message, chat_state, img_list = gradio_answer(chat_state, img_list, num_beams, temperature)
                    # print('=====================================================')
                    # print('Chat:', llm_message)
                    # print('=====================================================')
                    emotion, sentiment, humor = categorize(str(llm_message), str(emotion), str(sentiment), str(humor))
                    # print(f'emotion: {emotion}')
                    # print(f'sentiment: {sentiment}')
                    # print(f'humor: {humor}')
                    prompt_mid = ''
                    if emotion == '':
                        prompt_mid += 'emotion'
                    if sentiment == '':
                        if prompt_mid != '':
                            prompt_mid += ' and '
                        prompt_mid += 'sentiment'
                    if humor == '':
                        if prompt_mid != '':
                            prompt_mid += ' and '
                        prompt_mid += 'humor'
                    prompt = f"{prompt1}{prompt_mid}{prompt2}"

                filtered_data.at[i, 'chat'] = llm_message
                filtered_data.at[i, 'emotion'] = ('' if len(emotion) == 0 else emotion)
                filtered_data.at[i, 'sentiment'] = ('' if len(sentiment) == 0 else sentiment)
                filtered_data.at[i, 'humor'] = ('' if len(humor) == 0 else humor)
                # if '*' not in llm_message and counter == 5:
                # if ':' not in llm_message and '-' not in llm_message and counter == 3:
                if len(emotion) == 0 or len(sentiment) == 0 or len(humor) == 0:
                    filtered_data.at[i, 'done'] = 'X'
                else:
                    filtered_data.at[i, 'done'] = 'O'
                progress.update(1)
            else:
                progress.update(1)
                continue
            if i % 5 == 0:
                if 'Oxford' in insData :
                    filtered_data.to_csv(f'../../Data/Oxford_HIC/Minigpt4_{insData}.csv', index=False)
                else:
                    filtered_data.to_csv(f'../../Data/Instagram/Minigpt4_{insData}.csv', index=False)
    if 'Oxford' in insData :
        filtered_data.to_csv(f'../../Data/Oxford_HIC/Minigpt4_{insData}.csv', index=False)
    else:
        filtered_data.to_csv(f'../../Data/Instagram/Minigpt4_{insData}.csv', index=False)
if __name__ == '__main__':
    main()