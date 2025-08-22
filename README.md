# GAMC - Adaptable Advertising Meme Caption Generation Model

## Data
### 1. Oxford_HIC
[![DOI](https://zenodo.org/badge/DOI/10.1109/ICCV51070.2023.01856.svg)](https://doi.org/10.1109/ICCV51070.2023.01856) 
[![GitHub](https://img.shields.io/badge/GitHub-runjiali--rl/Oxford_HIC-darkgreen)](https://github.com/runjiali-rl/Oxford_HIC)
[![Data](https://img.shields.io/badge/Data-Google_drive-red)](https://drive.google.com/drive/folders/1BDuUcMeaWrFD8TwgHLhFPkuAwmoHaVNQ)
### 2. Instagram
[![Code](https://img.shields.io/badge/Code-instagram.ipynb-green)](Data/Instagram/instagram.ipynb)
[![Data](https://img.shields.io/badge/Data-Directory-red)](../Data/Instagram/)
* `Original_File/Home_xxx.csv` >> Instagram Post in home page (text == img alt)
* `Original_File/Done_xxx.csv` >> Instagram Post in post page (text == caption)
* `CaptionID_xxx.csv` >> Full Data with humor score
* `Filter_xxx.csv` >> Remove missing images
* `Generate_xxx.csv` >> After caption augmentation
### 3. Emotion-Sentiment-Humor Description by MiniGPT4 
[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2304.10592.svg)](https://doi.org/10.48550/arXiv.2304.10592)
[![GitHub](https://img.shields.io/badge/GitHub-Vision--CAIR/MiniGPT--4-darkgreen)](https://github.com/Vision-CAIR/MiniGPT-4.git)
[![Code](https://img.shields.io/badge/Code-instagram.ipynb-green)](Citations/minigpt4/demo_getdata.py)

## Code
### GAMC 
* Step 0 : `Preprocess` 
  * Data reformat & Augmentation: 
    * [tool.ipynb](Main/ipynb_tools/tool.ipynb)
    * [test_model.ipynb](Main/ipynb_tools/test_model.ipynb)
  * Data parsing: 
    * [parse_oxford.py](Data/Oxford_HIC/parse_oxford.py)
    * [parse_ins.py](Data/Instagram/parse_ins.py)[oxford_hic_dataset.zip](Main/oxford_hic_dataset.zip) 
<br><br>
* Step 1 : `Main model` 
  * code: [oxford_train_BLEU_minigpt4.py](Main/oxford_train_BLEU_minigpt4.py)
  * checkpoint: [checkpoint-004.pt](Main/Model/final/GAMC/20250421_oxford_3000_only1_300_82_ESH_bert_cross_concat/checkpoint-004.pt)
<br><br>
* Step 2 : `Adaptation` 
  * code: [oxford_train_BLEU_adapter_minigpt4.py](Main/oxford_train_BLEU_adapter_minigpt4.py)
  * checkpoint-MC: [checkpoint-002.pt](Main/Model/final/GAMC/MC/100up_only200_lessNotFunImg_53_171_passlength_12_MC_/all/checkpoint-002.pt)
  * checkpoint-SD: [checkpoint-011.pt](Main/Model/final/GAMC/SD/100up_only200_lessNotFunImg_169_55_passlength_10_SD_/all/checkpoint-011.pt)
<br><br>
* Step 3 : `Test result`
  * code: [oxford_predict_minigpt4.py](Main/oxford_predict_minigpt4.py)
  * organization: [result.ipynb](Main/ipynb_tools/result.ipynb)
### Evaluation
* `Humor Score` >> [humor_score.py](Main/humor_score.py)
* `Benign Score` >> Vilio [![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2012.07788.svg)](https://doi.org/10.48550/arXiv.2012.07788) 
[![GitHub](https://img.shields.io/badge/GitHub-Muennighoff/vilio-darkgreen)](https://github.com/Muennighoff/vilio)
* `Fluency Score` >> Parrot [![GitHub](https://img.shields.io/badge/GitHub-PrithivirajDamodaran/Parrot_Paraphraser-darkgreen)](https://github.com/PrithivirajDamodaran/Parrot_Paraphraser)
* `Diversity Score` >> [result.ipynb](Main/ipynb_tools/result.ipynb) (cosine similarity of image caption pairs clip embeddings)

## Framework
![alt text](./framework_detail.png)

## Baseline
### 1. ClipCap
[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2111.09734.svg)](https://doi.org/10.48550/arXiv.2111.09734) 
[![GitHub](https://img.shields.io/badge/GitHub-rmokady/CLIP_prefix_caption-darkgreen)](https://github.com/rmokady/CLIP_prefix_caption.git)
### 2. BITA 
[![DOI](https://zenodo.org/badge/DOI/10.1109/TGRS.2024.3359316.svg)](https://doi.org/10.1109/TGRS.2024.3359316) 
[![GitHub](https://img.shields.io/badge/GitHub-yangcong356/BITA-darkgreen)](https://github.com/yangcong356/BITA.git) <br>
Note: Due to the import packages, BITA can only run on linux system.
