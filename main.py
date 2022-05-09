import torch
import torch.nn.functional as F

from data.dataloader import skf_data_loader, test_data_loader 
from data.post_process import post_processing
from utils.utils import  Argument
from Trainer import Trainer

from transformers import AutoTokenizer

def main(args):

  tokenizer=AutoTokenizer.from_pretrained(args.model_name)

  if args.train == 'True':
    # Get train dataloader
    train_data, encoder3 = post_processing(args)
    train_dataloader_list, eval_dataloader_list = skf_data_loader(train_data, tokenizer, args)
    
  else:
    # Get test dataloader
    test_data = post_processing(args)
    test_dataloader = test_data_loader(test_data, tokenizer, args)
    train_dataloader_list, eval_dataloader_list = None, None

  trainer=Trainer(train_dataloader_list, eval_dataloader_list, args)

  if args.train == 'True':
    trainer.train(0)
    trainer.train(1)
    trainer.train(2)
    trainer.train(3)
    trainer.train(4)

  else:    
    results = trainer.inference(test_dataloader)
    ensemble = [(f1+f2+f3+f4+f5)/5 for (f1,f2,f3,f4,f5) in zip(results[0],results[1],results[2],results[3],results[4])]
    ensemble_logits = F.softmax(torch.tensor(ensemble),dim=1)
    torch.save(ensemble_logits,'output/skf_roberta_ensemble_logits.pt')


    
if __name__ == '__main__':
    args = Argument().add_args()
    main(args)