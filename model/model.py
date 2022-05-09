import torch
from torch import nn

from transformers import AutoModel, AutoConfig

class FCLayer(nn.Module):
    # both attention dropout and fc dropout is 0.1 on Roberta: https://arxiv.org/pdf/1907.11692.pdf
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh() # roberta and electra both uses gelu whereas BERT used tanh

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.activation(x)
        return self.linear(x)


class HiBERT(nn.Module):
  def __init__(self, args):
    super(HiBERT, self).__init__()

    self.args=args
    config = AutoConfig.from_pretrained(self.args.model_name)
    self.backbone_model= AutoModel.from_pretrained(self.args.model_name)
    
    self.dropout_rate = self.args.dropout
    self.digit_1_labels = self.args.digit_1_class
    self.digit_2_labels = self.args.digit_2_class
    self.digit_3_labels = self.args.digit_3_class
    
    self.classifier_1=FCLayer(2*config.hidden_size, self.digit_1_labels, self.dropout_rate)
    self.classifier_2=FCLayer(self.digit_1_labels+2*config.hidden_size, self.digit_2_labels, self.dropout_rate)
    self.classifier_3=FCLayer(self.digit_1_labels+self.digit_2_labels+2*config.hidden_size, self.digit_3_labels, self.dropout_rate)

    self.fc_layer=FCLayer(config.hidden_size, 2*config.hidden_size, self.dropout_rate)

  def forward(self, input_ids, attention_mask):
    output=self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
    projection_output=self.fc_layer(output.pooler_output)

    logit_1=self.classifier_1(projection_output)

    logit_2=self.classifier_2(torch.cat([logit_1,projection_output],dim=-1))

    logit_3=self.classifier_3(torch.cat([logit_1,logit_2,projection_output],dim=-1))

    return logit_1, logit_2, logit_3