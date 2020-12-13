import torch
from torch.utils.data import DataLoader
from data_processor import DataProcessor
from model import BertClassifier 
from tqdm import tqdm

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.data_processor = DataProcessor("/Users/a5560648/workspace/tutor/data", max_len=config["max_len"])
        self.model = BertClassifier(config=config)

    def train(self):
        data_loader = DataLoader(self.data_processor.get_dataset(), batch_size=config["batch_size"], shuffle=True, drop_last=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"]) 
        loss_fn = torch.nn.functional.cross_entropy
        for epoch in range(self.config["epoch"]):
            with tqdm(total=len(data_loader)) as pbar:
                for input_ids, token_type_ids, attention_mask, labels in data_loader:
                    optimizer.zero_grad()
                    output = self.model(input_ids, token_type_ids, attention_mask)
                    loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)


if __name__ == "__main__":
    config ={
        "bert": "bert-base-chinese",
        "hidden_size": 768, 
        "num_of_labels": 4,
        "dropout_rate": 0.1,
        "max_len": 30,
        "epoch": 10,
        "lr": 0.0005,
        "batch_size": 4
    }
    trainer = Trainer(config)
    trainer.train()
    # label = trainer.predict("我想買新手機了")
