import os
from os.path import join as opj
from warnings import filterwarnings
filterwarnings('ignore')

import time
import copy
from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModel, AutoConfig
from transformers.optimization import AdamW

from utils.utils import flat_accuracy, EarlyStopping
from model.model import HiBERT


class Trainer:
  def __init__(self, train_dataloader_list, eval_dataloader_list, args):
    self.train_dataloader_list=train_dataloader_list
    self.eval_dataloader_list=eval_dataloader_list
    self.args=args
    
    self.model=HiBERT(args).to(args.device)

  def setting(self, model):
    self.optimizer = torch.optim.AdamW(model.parameters(),lr=self.args.learning_rate, 
                                        weight_decay=self.args.weight_decay)
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.arg.Tmax, eta_min=0)
    self.criterion = nn.CrossEntropyLoss()
    self.scaler = amp.GradScaler()
    self.early_stopping = EarlyStopping(patience=3, mode='max')
    self.best_eval_acc = 0

  def sum_writer(self, fold_num):
    writer = SummaryWriter(log_dir='log/HiBERT_RoBERTa/fold'+str(fold_num))

    return writer

  def evaluation(self, model, eval_dataloader, criterion, writer):
    # ========================================
    #               Validating
    # ========================================
    model.eval()
    
    total_eval_loss=0
    digit_3_loss=0

    total_eval_accuracy=0
    digit_3_accuarcy=0

    digit_3_f1=0
    total_eval_f1=0

    total_val_batch=len(eval_dataloader)

    print('')
    for i,batch in enumerate(eval_dataloader):
      input_ids = batch['input_ids'].to(self.args.device)
      attention_mask = batch['attention_mask'].to(self.args.device)
      label_1 = batch['label_1'].to(self.args.device)
      label_2 = batch['label_2'].to(self.args.device)
      label_3 = batch['label_3'].to(self.args.device)

      with torch.no_grad():
        logit_1, logit_2, logit_3 = model(input_ids=input_ids,attention_mask=attention_mask)

      l1_loss=criterion(logit_1, label_1)
      l2_loss=criterion(logit_2, label_2)
      l3_loss=criterion(logit_3, label_3)

      loss=l1_loss*0.1+l2_loss*0.2+l3_loss*0.7

      total_eval_loss += loss.item()
      digit_3_loss += l3_loss.item()

      acc_1, f1_1 = flat_accuracy(logit_1, label_1)
      acc_2, f1_2 = flat_accuracy(logit_2, label_2)
      acc_3, f1_3 = flat_accuracy(logit_3, label_3)
      
      total_eval_accuracy += 0.1*acc_1+0.2*acc_2+0.7*acc_3
      digit_3_accuarcy += acc_3
      total_eval_f1 += 0.1*f1_1+0.2*f1_2+0.7*f1_3
      digit_3_f1 += f1_3

      writer.add_scalars("Loss/Valid", {'total_eval_loss':total_eval_loss/(i+1),
                                       'digit_3_loss':digit_3_loss/(i+1)}, i+1)
      
      writer.add_scalars("Accuracy/Valid", {'total_eval_acc':total_eval_accuracy/(i+1),
                                           'digit_3_acc': digit_3_accuarcy/(i+1)}, i+1)
      
      writer.add_scalars("F1/Valid", {'total_eval_f1': total_eval_f1/(i+1),
                                     'digit_3_f1': digit_3_f1/(i+1)}, i+1)
      
      
      print(f"\rValidation Batch {i+1}/{total_val_batch} , validation loss: {total_eval_loss/(i+1):.4f} , digit_3_loss: {digit_3_loss/(i+1):.4f} , Acc: {total_eval_accuracy/(i+1):.4f} , F1: {total_eval_f1/(i+1):.4f} \
      , digit_3_acc: {digit_3_accuarcy/(i+1):.4f} , digit_3_F1: {digit_3_f1/(i+1):.4f} ", end='')
    print('')
    return total_eval_accuracy/(i+1)

  def train(self, fold_num):

      model=copy.deepcopy(self.model)
      self.setting(model)
      writer=self.sum_writer(fold_num)
      
      train_dataloader=self.train_dataloader_list[fold_num]
      eval_dataloader=self.eval_dataloader_list[fold_num]

      print(f'\n# Fold {fold_num}')
      for epoch_i in range(0, self.args.max_epochs):
        
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.args.max_epochs))
        model.train()
        t0 = time.time()
        total_train_loss = 0
        digit_3_loss = 0
        total_batch=len(train_dataloader)

        for i, batch in enumerate(train_dataloader):
          input_ids = batch['input_ids'].to(self.args.device)
          attention_mask = batch['attention_mask'].to(self.args.device)
          label_1 = batch['label_1'].to(self.args.device)
          label_2 = batch['label_2'].to(self.args.device)
          label_3 = batch['label_3'].to(self.args.device)

          self.optimizer.zero_grad()

          with amp.autocast():
            logit_1, logit_2, logit_3 = model(input_ids=input_ids,attention_mask=attention_mask)

          l1_loss=self.criterion(logit_1, label_1)
          l2_loss=self.criterion(logit_2, label_2)
          l3_loss=self.criterion(logit_3, label_3)

          loss=l1_loss*0.1+l2_loss*0.2+l3_loss*0.7    
            
          total_train_loss += loss.item()
          digit_3_loss += l3_loss.item()

          writer.add_scalars("Loss/Train", {'total_train_loss':total_train_loss/(i+1),
                                          'digit_3_loss':digit_3_loss/(i+1)}, i+1)

          self.scaler.scale(loss).backward()
          self.scaler.step(self.optimizer)
          self.scaler.update()
            
          training_time = time.time() - t0

          print(f"\rTotal Batch {i+1}/{total_batch} , elapsed time : {training_time/60:.2f}m , train_loss : {total_train_loss/(i+1):.4f} , digit_3_loss : {digit_3_loss/(i+1):.4f} ", end='')
          
          if (i+1)%3000==0:
            eval_acc=self.evaluation(model, eval_dataloader, self.criterion, writer)

            if self.early_stopping.step(torch.tensor(eval_acc)):
                break

            if eval_acc>self.best_eval_acc:
              model_name=f'HiBERT_RoBERTa/HiBERT_skf_fold{fold_num}_roberta_large.pt'
              print(f'Best eval_acc: {self.best_eval_acc:.4f} -> {eval_acc:.4f}, Model_saved ===> ', f'HiBERT_skf_fold{fold_num}_roberta_large.pt')
              print('')
              torch.save(model.state_dict(), opj(self.args.weight_path, model_name))
              self.best_eval_acc=eval_acc

          if i+1==total_batch:
            eval_acc=self.evaluation(model, eval_dataloader, self.criterion, writer)

            if self.early_stopping.step(torch.tensor(eval_acc)):
                break

            if eval_acc>self.best_eval_acc:
              model_name=f'HiBERT_RoBERTa/HiBERT_skf_fold{fold_num}_roberta_large.pt'
              print(f'Best eval_acc: {self.best_eval_acc:.4f} -> {eval_acc:.4f}, Model_saved ===> ', f'HiBERT_skf_fold{fold_num}_roberta_large.pt')
              print('')
              torch.save(model.state_dict(), opj(self.args.weight_path, model_name))
              self.best_eval_acc=eval_acc

        self.scheduler.step()
        print("")
      writer.close()


  def inference(self, test_dataloader):
    skf_result_3=[]

    total_test_batch=len(test_dataloader)
    model=copy.deepcopy(self.model)

    for i in range(0,5):

        model.load_state_dict(torch.load(opj(self.args.weight_path,'HiBERT_RoBERTa/HiBERT_skf_fold'+str(i)+'_roberta_large.pt')))
        print('')
        print('Model is successfully loaded: HiBERT_skf_fold'+str(i)+'_roberta_large.pt')

        model.eval()
        predictions_3=[]

        for i,batch in enumerate(test_dataloader):
          input_ids = batch['input_ids'].to(self.args.device)
          attention_mask = batch['attention_mask'].to(self.args.device)

          with torch.no_grad():
            logit_1, logit_2, logit_3 = model(input_ids=input_ids,attention_mask=attention_mask)

          pred_3=logit_3.cpu().numpy()

          predictions_3=[*predictions_3, *pred_3]

          print(f"\rTest Batch {i+1}/{total_test_batch} ", end='')

        skf_result_3.append(predictions_3)
        print('')

    return skf_result_3 