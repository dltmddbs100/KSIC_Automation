# KSIC_Automation
- Our natural language processing team([이승윤](https://github.com/dltmddbs100), [김일구](https://github.com/dlfrnaos19), [박범수](https://github.com/Cloud9Bumsu)) achieved 1st place in the 1st AI competition hosted by the National Statistical Office.
- This task is classification task using KSIC(한국표준산업분류) Dataset provided by the National Statistical Office.
- We should assine proper industry categories using natural language form.
- You can check the article about this competititon [here](https://data.kostat.go.kr/sbchome/contents/cntPage.do?cntntsId=CNTS_000000000000575&curMenuNo=OPT_09_03_00_0)


## Model Process

![프로세스 사진](https://user-images.githubusercontent.com/55730591/167529122-06e0e78e-ffea-493b-80c9-2ea6f77c2c2c.jpg)

- The whole process consists of data cleansing, model building, and ensemble.


### Pre-processing

+ **Refine special characters, patterns, and typos**
    + All variables consist of natural language form
    + It has lots of patterns and typos
    + Use Regex to refine these all

+ **Separation and spacing of key words**
    + Lots of typos and missing of spacing yield bad influence in tokenizing
    + We make keywords for each category which seem to be important to classifiaction
    + Forced spacing using Regex and [QuickSpacer](https://github.com/cosmoquester/quickspacer) library
    + This makes huge increase in model performance

### Modeling
+ **Pre-training Models**
    + [klue/roberta-large](https://huggingface.co/klue/roberta-large)
    + Tried various models like electra, bert, bigbird but roberta was best
    + Based on this pre-training model, we build to different models

+ **HiRoberta**
    + The data provided in the competition is classified into main, middle, and sub categories according to the Korean standard industry classification
    + It is a hierarchical classification problem that has dependence rather than an independent relationship between labels.
    + HiRoberta inherits information used to predict main and middle classes when predicting subcategories
    + The structure can use parent categories information in subcategory clasification, so that hierarchical characteristics can be efficiently conveyed to the model

+ **Flat Roberta**
    + Simple linear classification model only focus on subcategories
    + Roberta - linear layer - softmax 
    + Using TPU in Google Colab with high speed


### Training & Inference Strategy

| Model | HiRoberta | Flat Roberta |
| --- | --- | --- |
| CPU | Intel(R) Xeon(R) CPU | Intel(R) Xeon(R) CPU |
| Tensor | Colab P100 | Colab TPU V2 |
| Training latency(5-folds)               | 36 hours | 2.5 hours |
| Inference latency(5-folds)            | 20 min | 25min |
| Pre-trained model | klue/roberta-large | klue/roberta-large |
| Inference latency            | 20 min | 25min |
| Epoch | 4 | 4 |
| Batch size | 64 | 64(Global batch 512) |
| Learning rate | 2e-5, CosineAnnealing(T=4) | [3e-5, 2e-5, le-5, 5e-6] |
| Optimizer | AdamW(weight_decay=1e-2) | Adam |
| Loss | Custom CrossEntropy | CrossEntropy |
| Callback | EarlyStop(Acc max) | EarlyStop(Val loss min) |


+ In HiRoberta Transform the composition of the loss function in consideration of the designed model characteristics and the competition evaluation formula (main: middle: sub = 1:2:7)
+ Due to imbalanced labels, apply Stratified K-fold strategy and ensemble
+ Finally, ensemble two different 5-fold models and make inference



