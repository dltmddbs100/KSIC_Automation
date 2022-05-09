from warnings import filterwarnings
filterwarnings('ignore')

import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, dataset, tok, infer=False):
        super().__init__()
        self.tok = tok
        self.dataset = dataset
        self.infer = infer
    
    def __getitem__(self, idx):
      if not self.infer:
        instance = self.dataset['text'][idx]
        digit_1 = self.dataset['label_1'][idx]
        digit_2 = self.dataset['label_2'][idx]
        digit_3 = self.dataset['label_3'][idx]

        input_input_attn = self.tok(instance, max_length=55, truncation=True)

        return {'input_ids': torch.tensor(input_input_attn['input_ids'],dtype=torch.long),
                'attention_mask':torch.tensor(input_input_attn['attention_mask']).int(),
                'digit_1': torch.tensor(digit_1).unsqueeze(0),
                'digit_2': torch.tensor(digit_2).unsqueeze(0),
                'digit_3': torch.tensor(digit_3).unsqueeze(0)}

      else:
        instance = self.dataset['text'][idx]
        input_input_attn = self.tok(instance, padding=True)
        
        return {'input_ids': torch.tensor(input_input_attn['input_ids'],dtype=torch.long),
                'attention_mask':torch.tensor(input_input_attn['attention_mask']).int()}
    
    def __len__(self):
        return len(self.dataset)

def collate_fn_padd_train(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''

    input_ids = [ t['input_ids'] for t in batch ]
    attention_mask = [ t['attention_mask'] for t in batch ]
    label_1 = [ t['digit_1'] for t in batch ]
    label_2 = [ t['digit_2'] for t in batch ]
    label_3 = [ t['digit_3'] for t in batch ]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,padding_value=1,batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask,batch_first=True)

    return {'input_ids': input_ids,
            'attention_mask':attention_mask,
            'label_1': torch.tensor(label_1),
            'label_2': torch.tensor(label_2),
            'label_3': torch.tensor(label_3)}

def collate_fn_padd_test(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''

    input_ids = [ t['input_ids'] for t in batch ]
    attention_mask = [ t['attention_mask'] for t in batch ]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,padding_value=1,batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask,batch_first=True)

    return {'input_ids': input_ids,
            'attention_mask':attention_mask}