import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_bert import BertModel

# Keyword Parameters
# Positional Parameters

class BertClassifier(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["bert"])
        self.classifier_layer = nn.Linear(config["hidden_size"], config["num_of_labels"])
        self.dropout = nn.Dropout(p=config["dropout_rate"])

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        cls_output = output[0][:, 0, :]
        out = self.classifier_layer(cls_output)
        return out




