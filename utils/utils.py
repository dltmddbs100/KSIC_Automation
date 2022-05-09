from argparse import ArgumentParser

import torch
from sklearn.metrics import f1_score


def flat_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, axis=1)
    acc=torch.sum(pred_flat == labels) / len(labels)
    f1=f1_score(labels.cpu(), pred_flat.cpu(), average='macro')
    return acc, f1


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class Argument():
  def __init__(self):
    self.parser=ArgumentParser()

  def add_args(self):

    self.parser.add_argument('--train', type=str, default='True')
    self.parser.add_argument('--model_name', type=str, default='klue/roberta-large')
    self.parser.add_argument('--weight_path', type=str, default='weights/')
    self.parser.add_argument('--sub_path', type=str, default='sub/')
    self.parser.add_argument('--path_to_train_data', type=str, default='data/train_final_spacing.csv')
    self.parser.add_argument('--path_to_test_data', type=str, default='data/test_final_spacing.csv')
    self.parser.add_argument('--device', type=str, default='cuda')
    self.parser.add_argument('--batch_size', type=int, default='64')
    self.parser.add_argument('--max_epochs', type=int, default='4')
    self.parser.add_argument('--max_len', type=int, default='55')
    self.parser.add_argument('--learning_rate', type=float, default='2e-5')
    self.parser.add_argument('--weight_decay', type=float, default='1e-2')
    self.parser.add_argument('--Tmax', type=int, default='4')
    self.parser.add_argument('--dropout', type=float, default='0.1')
    self.parser.add_argument('--digit_1_class',type=int, default='19')
    self.parser.add_argument('--digit_2_class',type=int, default='74')
    self.parser.add_argument('--digit_3_class',type=int, default='225')
    args = self.parser.parse_args()
   
    self.print_args(args)
    
    return args
    
  def print_args(self, args):
    print('====== Input arguments ======')    
    
    for idx, (key, value) in enumerate(args.__dict__.items()):
        if idx == 0:print("argparse{\n", "\t", key, ":", value)
        elif idx == len(args.__dict__) - 1:print("\t", key, ":", value, "\n}")
        else:print("\t", key, ":", value)