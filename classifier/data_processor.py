import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import csv


class MyDataset(Dataset):

    def __init__(self, input_ids, token_type_ids, attention_mask, targets):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.token_type_ids[index]), torch.tensor(self.attention_mask[index]), torch.tensor(self.targets[index])

corpus_name = "ptt-classification-corpus.csv"


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

    def _load_data(self):
        filename = os.path.join(self.data_dir, corpus_name)
        if not os.path.exists(filename):
            raise FileNotFoundError("The directory should contain the data named {}".format(corpus_name))
        with open(filename, 'r') as file:
            text = file.read()
            lines = text.split('\n')

            features = []
            targets = []
            for line in lines:
                if not line:
                    continue
                entries = line.rsplit(',', 1)
                features.append(entries[0])
                targets.append(entries[1])

            classes = list(set(targets))
            target_dict = {}
            for idx, class_name in enumerate(classes):
                target_dict[class_name] = idx

            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            out = tokenizer(features, padding="max_length", max_length=self.max_len, truncation=True)
            input_ids = out["input_ids"]
            token_type_ids = out["token_type_ids"]
            attention_mask = out["attention_mask"]
            # for feature in features:
                # ids = []
                # for char in feature:
                    # if char in self.vocab:
                        # ids.append(self.vocab[char])
                # if len(ids) < self.max_len:
                    # ids.extend([0] * (self.max_len-len(ids)))
                # else:
                    # ids = ids[:self.max_len]
                # input_ids.append(ids)

            target_ids = []
            for target in targets:
                target_id = target_dict[target]
                target_ids.append(target_id)

            assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(target_ids)

        return input_ids, token_type_ids, attention_mask, target_ids

    def get_dataset(self):
        return MyDataset(input_ids=self.data[0], token_type_ids=self.data[1], attention_mask=self.data[2], targets=self.data[3]) 

if __name__ == "__main__":
    data_processor = DataProcessor("/Users/a5560648/workspace/tutor/data")
    print(data_processor.get_dataset())


