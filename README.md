# Data
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

# Code
### GAMC [![DOI](https://zenodo.org/badge/DOI/10.6342/NTU202501318.svg)](https://doi.org/10.6342/NTU202501318)
* Step 1 : `Main model` [oxford_train_BLEU_minigpt4.py](Main/oxford_train_BLEU_minigpt4.py)
* Step 2 : `Adaptation` [oxford_train_BLEU_adapter_minigpt4.py](Main/oxford_train_BLEU_adapter_minigpt4.py)
* Step 3 : `Test result` [oxford_predict_minigpt4.py](Main/oxford_predict_minigpt4.py)
### Evaluation
* `Humor Score` >> [humor_score.py](Main/humor_score.py)
* `Benign Score` >> Vilio [![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2012.07788.svg)](https://doi.org/10.48550/arXiv.2012.07788) 
[![GitHub](https://img.shields.io/badge/GitHub-Muennighoff/vilio-darkgreen)](https://github.com/Muennighoff/vilio)
* `Fluency score` >> Parrot [![GitHub](https://img.shields.io/badge/GitHub-PrithivirajDamodaran/Parrot_Paraphraser-darkgreen)](https://github.com/PrithivirajDamodaran/Parrot_Paraphraser)
* `Diversity score` >> cosine similarity of image caption pairs clip embeddings

# Baseline
### 1. ClipCap
[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2111.09734.svg)](https://doi.org/10.48550/arXiv.2111.09734) 
[![GitHub](https://img.shields.io/badge/GitHub-rmokady/CLIP_prefix_caption-darkgreen)](https://github.com/rmokady/CLIP_prefix_caption.git)
### 2. BITA
[![DOI](https://zenodo.org/badge/DOI/10.1109/TGRS.2024.3359316.svg)](https://doi.org/10.1109/TGRS.2024.3359316) 
[![GitHub](https://img.shields.io/badge/GitHub-yangcong356/BITA-darkgreen)](https://github.com/yangcong356/BITA.git)

## Download repo
### Install coco-caption
#### 1. Install the model

[Data](#my-custom-anchor-point)   

- [x] #739
- [ ] https://github.com/octo-org/octo-repo/issues/740
- [ ] Add delight to the experience when all tasks are complete :tada:
To properly obtain the CIDE
> To properly obtain the CIDE
>
The background color is `#ffffff` for light mode and `#000000` for dark mode.
### dataset
3. Set the environment variable in `~/.bashrc` by adding the following lines
[Rico](https://interactionmining.org/rico) / [Rico UI Screenshots and View Hierarchies dataset](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz)

Screen2Words: [paper](https://arxiv.org/abs/2108.03353) / [code](https://github.com/google-research/google-research/tree/master/screen2words) / [dataset](https://github.com/google-research-datasets/screen2words)
```
git clone --recursive https://github.com/RainYuGG/image-captioning-based-on-Screen2Words.git
```
* Install [coco-caption](#install-coco-caption) for evaluation (BLEU, CIDEr).
## Build Environment & Requirement
* ```adapter_type: "vit"``` : vit adapter ("vit", "vit_grayscale")
* Use conda to build the environment
```
conda env create -f environment.yml
```
[Install the model](#Data)