import os
from os.path import join as opj
from warnings import filterwarnings
filterwarnings('ignore')

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold

from data.dataset import BERTDataset, collate_fn_padd_train, collate_fn_padd_test


def skf_data_loader(train, tokenizer, args):
  train_dataset=BERTDataset(train, tokenizer)

  kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=1514)

  fold_train_idx=[]
  fold_eval_idx=[]

  for train_idx, eval_idx in kfold.split(train,train['label_3']):
    fold_train_idx.append(train_idx)
    fold_eval_idx.append(eval_idx)


  # K-fold Cross Validation model evaluation
  train_dataloader_list=[]
  eval_dataloader_list=[]

  for i, (train_ids, test_ids) in enumerate(zip(fold_train_idx,fold_eval_idx)):
      
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    eval_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
      
    # Define data loaders for training and testing data in this fold
    tr_loader=DataLoader(train_dataset, batch_size=args.batch_size,
                        sampler=train_subsampler,collate_fn=collate_fn_padd_train,
                        num_workers=4, pin_memory=True)
    evl_loader = DataLoader(train_dataset,batch_size=args.batch_size,
                          sampler=eval_subsampler,collate_fn=collate_fn_padd_train,
                          num_workers=4, pin_memory=True)
    
    train_dataloader_list.append(tr_loader)
    eval_dataloader_list.append(evl_loader) 


  print(train_dataloader_list)
  
  return train_dataloader_list, eval_dataloader_list


def test_data_loader(test,tokenizer,args):
  test_dataset=BERTDataset(test, tokenizer, infer=True)

  test_dataloader = DataLoader(test_dataset,
                              batch_size=args.batch_size, 
                              shuffle=False,
                              pin_memory=True,
                              collate_fn=collate_fn_padd_test)

  return test_dataloader