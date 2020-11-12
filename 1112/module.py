import torch
import torch.nn as nn
import torch.nn.functional as F

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
            self.fc = nn.Linear(4 * 256, 1024)

            # Linear
            # 1024 x 5
            self.classifier = nn.Linear(1024, 5)

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


if __name__ == "__main__":
    simple_model = SimpleModel()
    print(simple_model)

    x = [2, 7, 200, 4000]
    x = torch.tensor([x])

    print(type(x), x.dtype)
    output = simple_model(x)

    print(output)


