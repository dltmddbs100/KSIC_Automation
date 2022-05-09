import os
from os.path import join as opj

import re
import time
from tqdm import tqdm

import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from quickspacer import Spacer


def word_preprocess(data, column):

  for i in tqdm(range(len(data)),total=len(data)):
    try:
      process=re.sub('([0-9]+)m\^{0,1}2',r'\1㎡ ',data[column][i])
      process=re.sub('평방미터|제곱미터|미터제곱|제곱|평방|미터','㎡ ',process)
      process=re.sub('http[s:]*//[a-zA-Z0-9-]+\.[a-zA-Z0-9-_\.?=]+','온라인',process)
      process=re.sub('[？\?@:\"]+','',process)
      process=re.sub('([가-힣])[\'`]+',r'\1 ',process)
      process=re.sub('\.{2,}','.',process)
      process=re.sub('[ㄱ-ㅎㅏ-ㅣ]','',process)

      process=re.sub('([실부집정지민점량교업서물차년사체장원층동생이인님객자주들을를]{1})대상',r'\1 대상',process)
      process=re.sub('([소도판]+)매',r' \1매',process)
      process=re.sub('미만',' 미만',process)
      process=re.sub('및',' 및 ',process)
      process=re.sub('(교습|강습|수강|교육|학원|방문|과외|강사|지원)',r' \1',process)
      process=re.sub('(자동차|제조|부[속]*품|제작|조립|가공|절삭|정비|수리|소비자|서비스|고객|제공|운송|갖추)',r' \1',process)
      process=re.sub('(잡화|위주[로의]*|기타|의류|의복|세탁|수선)',r' \1 ',process)
      process=re.sub('(산업[용품|용|단지]*)([사용자]*)',r' \1 \2',process)
      process=re.sub('([기|원|농|폐|부]{0,1}자재)',r' \1',process)
      process=re.sub('전 +기자재',' 전기 자재',process)
      process=re.sub('전 자재료','전자 재료',process)
      process=re.sub(' +',' ',process)
      process=process.strip()
      data.at[i,column]=process
    except:
      pass

  return data    


def spacing_data(data, column, batch):
  
  datas=data.copy()
  datas=datas.fillna('[NAN]')

  spacer = Spacer(level=3)
  spacing=spacer.space(datas[column].tolist(),batch_size=batch)
  
  datas[column]=spacing

  return datas


def word_final_process(data, column):

  for i in tqdm(range(len(data)),total=len(data)):
    try:
      process=re.sub('제 공','제공',data[column][i])
      process=re.sub('제 작','제작',process)
      process=re.sub(' 제 조',' 제조',process)
      process=re.sub('절 단','절단',process)
      process=re.sub('장 비','장비',process)
      process=re.sub('가 공','가공',process)
      process=re.sub('영 어','영어',process)
      process=re.sub('판 매','판매',process)
      process=re.sub('소 매','소매',process)
      process=re.sub('도 매','도매',process)
      process=re.sub(' 합 판','합판',process)
      process=re.sub('실 내',' 실내',process)
      process=re.sub('특장 차','특장차',process)
      process=re.sub(' +',' ',process)
      process=process.strip()
      data.at[i,column]=process
    except:
      pass

  return data    


def preprocess(data, tokenizer, mode):
  data_clean=data.copy()

  data_clean=word_preprocess(data_clean,'text_obj')
  data_clean=word_preprocess(data_clean,'text_mthd')
  data_clean=word_preprocess(data_clean,'text_deal')

  if mode=='train':
    data_clean=data_clean.drop_duplicates(['text_obj','text_mthd','text_deal','digit_3']).reset_index(drop=True)

  data_clean=spacing_data(data_clean, 'text_obj', 128)
  data_clean=spacing_data(data_clean, 'text_mthd', 128)
  data_clean=spacing_data(data_clean, 'text_deal', 128)

  data_clean=word_final_process(data_clean, 'text_obj')
  data_clean=word_final_process(data_clean, 'text_mthd')
  data_clean=word_final_process(data_clean, 'text_deal')


  if mode=='train':
      data_clean=data_clean.fillna('[NAN]')
      data_clean['text']=data_clean['text_obj']+'-'+data_clean['text_mthd']+'-'+data_clean['text_deal']

      text_tokenize=[]
      for i in tqdm(data_clean['text']):
        text_tokenize.append(tokenizer.encode(i))

      data_clean['text_tokenize']=text_tokenize
      data_clean['text_tokenize']=data_clean['text_tokenize'].apply(str)

      data_clean=data_clean.drop_duplicates(['text_tokenize','digit_3']).reset_index(drop=True)
      data_clean=data_clean.drop(['text','text_tokenize'],axis=1)
  

      data_clean.to_csv(opj('data/','train_final_spacing.csv'),index=False)
      print('\n=> Train Data saved at train_final_spacing.csv')

  else:
      data_clean.to_csv(opj('data/','test_final_spacing.csv'),index=False)
      print('\n=> Test Data saved at test_final_spacing.csv')


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    train=pd.read_csv(opj('data/', '1. 실습용자료.txt'),encoding='euc-kr',sep='|')
    test=pd.read_csv(opj('data/', '2. 모델개발용자료.txt'),encoding='euc-kr',sep='|')
    print('\nPreprocessing Train Data...')
    preprocess(train,tokenizer,mode='train')
    print('\nPreprocessing Test Data...')
    preprocess(test,tokenizer,mode='test')
