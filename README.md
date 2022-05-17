# KSIC_Automation
-  **1st / 395 Teams** :1st_place_medal:
- **Our natural language processing team([이승윤](https://github.com/dltmddbs100), [김일구](https://github.com/dlfrnaos19), [박범수](https://github.com/Cloud9Bumsu)) achieved 1st place in the 1st AI competition hosted by the National Statistical Office**
- This is a classification task using KSIC(한국표준산업분류) Dataset provided by the National Statistical Office
- We should assine proper industry categories using natural language form
- You can check the article about this competititon [here](https://data.kostat.go.kr/sbchome/contents/cntPage.do?cntntsId=CNTS_000000000000575&curMenuNo=OPT_09_03_00_0)


## Model Process
- The whole process consists of data cleansing, model building, and ensemble

![프로세스 사진](https://user-images.githubusercontent.com/55730591/167529122-06e0e78e-ffea-493b-80c9-2ea6f77c2c2c.jpg)


## Pre-processing

+ **Refine special characters, patterns, and typos**
    + All variables consist of natural language form
    + It has lots of patterns and typos
    + Use Regex to refine these all

+ **Separation and spacing of key words**
    + Lots of typos and missing of spacing yield bad influence in tokenizing
    + We make keywords for each category which seem to be important to classifiaction
    + Forced spacing using Regex and [QuickSpacer](https://github.com/cosmoquester/quickspacer) library
    + This makes huge increase in model performance

## Modeling
+ **Pre-training Models**
    + [klue/roberta-large](https://huggingface.co/klue/roberta-large)
    + Tried various models like electra, bert, bigbird but roberta was best
    + Based on this pre-training model, we build to different models

+ **HiRoBERTa**
    + The data provided in the competition is classified into main, middle, and sub categories according to the Korean Standard Industry Classification
    + It is a hierarchical classification problem that has dependence rather than an independent relationship between labels
    + HiRoBERTa inherits information used to predict main and middle classes when predicting subcategories
    + The structure can use parent categories information in subcategory clasification, so that hierarchical characteristics can be efficiently conveyed to the model

+ **Flat RoBERTa**
    + Simple linear classification model only focus on subcategories
    + RoBERTa - linear layer - softmax 
    + Using TPU in Google Colab with high speed


## Training & Inference Strategy

| Model | HiRoBERTa | Flat RoBERTa |
| --- | --- | --- |
| CPU | Intel(R) Xeon(R) CPU | Intel(R) Xeon(R) CPU |
| Tensor | Colab P100 | Colab TPU V2 |
| Training latency(5-folds)               | 36 hours | 2.5 hours |
| Inference latency(5-folds)            | 20 min | 25 min |
| Pre-trained model | klue/roberta-large | klue/roberta-large |
| Epoch | 4 | 4 |
| Batch size | 64 | 64(Global batch 512) |
| Learning rate | 2e-5, CosineAnnealing(T=4) | [3e-5, 2e-5, le-5, 5e-6] |
| Optimizer | AdamW(weight_decay=1e-2) | Adam |
| Loss | Custom CrossEntropy | CrossEntropy |
| Callback | EarlyStop(Acc max) | EarlyStop(Val loss min) |

+ In HiRoBERTa, we transform the composition of the loss function in consideration of the designed model characteristics and the competition evaluation formula (main: middle: sub = 1:2:7)
+ Due to imbalanced labels, apply Stratified K-fold strategy and ensemble
+ Finally, ensemble two different 5-fold models and make inference


## Installation
```
!https://github.com/dltmddbs100/KSIC_Automation.git
!pip install transformers
!pip install tensorboard
!pip install quickspacer
cd /content/SimCSE/
```

## Getting Started HiRoBERTa
**Make dataset to 'data' directory from raw data.** <br/>
Raw datasets are not provided in this repository due to security issues. <br/>
:exclamation: Caution : Only those who have participated in the contest and have data can run it.
```python
# Make processed datasets
# This code will yeild 'train_final_spacing' and 'test_final_spacing'
!python data/preprocess.py
```

## Training HiRoBERTa Model
Training 5-fold models <br/>
```python
# 2. Train
!python main.py --train 'True'
                --model_name : 'klue/roberta-large'
                --weight_path : 'weights/'
                --sub_path : 'sub/'
                --path_to_train_data : 'data/train_final_spacing.csv'
                --path_to_test_data : 'data/test_final_spacing.csv'
                --device : 'cuda'
                --batch_size : 64
                --max_epochs : 4
                --max_len : 35
                --learning_rate : 2e-05
                --weight_decay : 0.01
                --dropout : 0.1
                --Tmax : 4
                --digit_1_class : 19
                --digit_2_class : 74
                --digit_3_class : 225 
```
+ `--train`: If you want to train the model, it should be 'True' while test argument is 'False'.
+ `--model_name`: The name or path of a transformers-based pre-trained checkpoint (default: klue/bert-base)
+ `--weight_path`: The place where your trained weights are saved.
+ `--device`: Supports 'cuda' or 'cpu'.
+ `--Tmax`: Maximum number of iterations in CosineAnnealingLR. 
+ `--digit_1_class`: Number of the main categories label.
+ `--digit_2_class`: Number of the middle categories label.
+ `--digit_3_class`: Number of the sub categories label.


## Training HiRoBERTa Model
Inference with 5-fold models, make and save average ensemble logits <br/>
Finally, startified ensemble logit file is saved in 'output' directory <br/>
```python
# 3.Inference
!python main.py --train 'False'
```

## Training Flat RoBERTa & Final Ensemble Inference
You can train standard RoBERTa model in `TF_roberta_large_skf&ensemble.ipynb` <br/>
At the end of this code, it loads HiRoBERTa's logits and ensemble with Flat RoBERTa which leads to submission file <br/>
By considering the hierarchical and horizontal characteristics of dataset at the same time, model can have a multi-faceted perspective in classification
