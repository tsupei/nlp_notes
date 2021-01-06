import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig
from transformers.modeling_bert import BertForQuestionAnswering
from torch.utils.data import DataLoader
from data_processor import DataProcessor

class Trainer():

    def __init__(self):
        self.config = BertConfig("bert-base-chinese")
        self.model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
        self.data_processor = DataProcessor(max_len=self.config.max_position_embeddings)

    def train(self, num_of_epochs):
        dataset = self.data_processor.get_dataset()
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4)

        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(num_of_epochs):
            for data in data_loader:
                optimizer.zero_grad()
                output = self.model(
                        input_ids = data[0],
                        token_type_ids = data[1], 
                        attention_mask = data[2], 
                        start_positions = data[3], 
                        end_positions = data[4]
                )
                output[0].backward()
                optimizer.step()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(1)
    
    


