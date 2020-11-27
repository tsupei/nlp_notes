import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
from tqdm import tqdm

# tensor
# t = torch.tensor([[1,2,3], [1, 2,3]], dtype=torch.float32)
# print(t.shape)

class SimpleModel(nn.Module):

        def __init__(self):
            super().__init__()
            # 詞嵌入
            vocabulary_size = 5000
            self.embedding_layer = nn.Embedding(vocabulary_size, 256, padding_idx=0)

            # 4, 256
            # 4 x 256

            # Fully connected
            # Linear Layer
            # 256 X 1024
            self.fc = nn.Linear(30 * 256, 1024)

            # Linear
            # 1024 x 5
            self.classifier = nn.Linear(1024, 4)

        def forward(self, x):
            # x = [2, 17, 200, 4000]
            # tensor
            x = self.embedding_layer(x)
            origin_shape = x.shape # 
            # print(x.size())
            x = x.view(origin_shape[0], origin_shape[1] * origin_shape[2]) # [batch, sequence length, embedding_size] -> [batch, seq * embed_size] 
            # x = x.view(x.size(0), x.size(1) * x.size(2))
            x = F.relu(x)
            x = self.fc(x)
            x = F.relu(x)
            x = self.classifier(x)
            return x


class MyDataset(Dataset):

    def __init__(self, data):
        self.input_ids = data[0]
        self.target_ids = data[1]
        assert len(self.input_ids) == len(self.target_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.target_ids[index])

# txt
# csv
# json
# pickle

def load_csv_data():
    filename = "./data/ptt-classification-corpus.csv"
    with open(filename, 'r') as file:
        rows = csv.reader(file)
        for row in rows:
            print(row)

def load_data():
    filename = "./data/ptt-classification-corpus.csv"

    vocab = {}
    with open("./vocab.txt", 'r') as file:
        text = file.read()
        lines = text.split('\n')
        lines = lines[:-1]
        for idx, line in enumerate(lines):
            vocab[line] = idx

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

        input_ids = []
        for feature in features:
            ids = []
            for char in feature:
                if char in vocab:
                    ids.append(vocab[char])
            max_len = 30
            if len(ids) < max_len:
                # Fill in 
                ids.extend([0] * (max_len-len(ids)))
            else:
                ids = ids[:max_len]
            input_ids.append(ids)
        
        target_ids = []
        for target in targets:
            target_id = target_dict[target]
            target_ids.append(target_id)

        # for i in range(50):
            # print(input_ids[i], target_ids[i])
        return input_ids, target_ids


if __name__ == "__main__":
    # data = load_csv_data()
    data = load_data()
    # input_ids, target_ids = load_data()

    dataset = MyDataset(data)
    print(len(dataset))
    data_loader = DataLoader(dataset, 4, shuffle=True, drop_last=True)

    model = SimpleModel()
    # loss_function = F.cross_entropy
    # print(model)
    # print(list(model.named_parameters()))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    for epoch in range(1000):
        correct = 0
        avg_loss = 0
        with tqdm(total=len(data_loader)) as progress_bar:
            for sample in data_loader:
                optimizer.zero_grad()
                features, targets = sample
                # print(features.shape, targets.shape)
                output = model(features)
                predicts = output.argmax(dim=-1)
                # print(predicts)
                # print(targets)
                # raise
                
                x = list(model.parameters())[0]
                loss = F.cross_entropy(output, targets)
                avg_loss += loss

                for i in range(4):
                    if predicts[i] == targets[i]:
                        correct += 1
                progress_bar.update(1)

        acc = correct / (len(data_loader) * 4)
        avg_loss /= len(data_loader)
        print("Loss: {} Acc: {}".format(avg_loss, acc))

        loss.backward()
        optimizer.step()



    # simple_model = SimpleModel()
    # print(simple_model)
    # x = [2, 7, 200, 4000]
    # x = torch.tensor([x])
    # print(type(x), x.dtype)
    # output = simple_model(x)
    # # output = F.softmax(output)
    # print(output)


