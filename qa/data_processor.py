import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json 


class MyDataset(Dataset):

    def __init__(self, input_ids, token_type_ids, attention_mask, start_positions, end_positions):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_positions = start_positions
        self.end_positions = end_positions

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.token_type_ids[index]), torch.tensor(self.attention_mask[index]), torch.tensor(self.start_positions[index]), torch.tensor(self.end_positions[index])


class DataProcessor(object):
    def __init__(self, data_dir, max_len=30):
        self.data_dir = data_dir
        self.vocab = self._load_vocab()
        self.max_len = max_len
        self.data = self._load_data()

    def _load_vocab(self):
        filename = os.path.join(self.data_dir, "vocab.txt")
        if not os.path.exists(filename):
            raise FileNotFoundError("The directory should contain the data named {}".format("vocab.txt"))
        vocab = {}
        with open(filename, 'r') as file:
            text = file.read()
            lines = text.split("\n")
            for idx, line in enumerate(lines):
                vocab[line] = idx
        return vocab

    def _load_json_data(self):
        filename = "./DRCD-master/DRCD_dev.json"
        with open(filename, 'r', encoding="utf8") as file:
            data = json.load(file)
            # data is a dict
            samples = []
            for item in data["data"]:
                for paragraph in item["paragraphs"]:
                    for qa in paragraph["qas"]:
                        answer = qa["answers"][0]
                        sample = {
                            "context": paragraph["context"],
                            "question": qa["question"],
                            "start_position": answer["answer_start"],
                            "end_position": answer["answer_start"] + len(answer["text"])
                        }
                        samples.append(sample)
            return samples

    def _load_data(self):
        samples = self._load_json_data()

        start_positions = []
        end_positions = []

        features = []
        for sample in samples:
            features.append((sample["context"], sample["question"]))
            start_positions.append(sample["start_position"])
            end_positions.append(sample["end_position"])

        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        out = tokenizer(features, padding="max_length", max_length=self.max_len, truncation=True)
        input_ids = out["input_ids"]
        token_type_ids = out["token_type_ids"]
        attention_mask = out["attention_mask"]
        return input_ids, token_type_ids, attention_mask, start_positions, end_positions

        # assert len(input_ids) == len(token_type_ids) == len(attention_mask) 
        # return input_ids, token_type_ids, attention_mask, target_ids

    def get_dataset(self):
        return MyDataset(input_ids=self.data[0], token_type_ids=self.data[1], attention_mask=self.data[2], start_positions=self.data[3], end_position=self.data[4]) 

if __name__ == "__main__":
    data_processor = DataProcessor("/Users/a5560648/workspace/tutor/data")
    print(data_processor.get_dataset())


