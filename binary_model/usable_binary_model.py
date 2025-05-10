import torch
from dataset.data_fetch import UCIDataset
from binary_model.binary_mlp import MLP as Binary_MLP, train_model
from torch.utils.data import DataLoader, random_split

data = UCIDataset(binary=True)

class Binary_model():
    def __init__(self):
       self.model = Binary_MLP(num_features = 13, dropout=0.20)
       self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
       self.loss_fn = torch.nn.CrossEntropyLoss()

       len_dataset = len(data)
       train_size = int(0.7 * len_dataset)
       test_size = len_dataset - train_size

       self.train_dataset, self.test_dataset = random_split(data, [train_size, test_size])

       self.train_loader = DataLoader(
       dataset=self.train_dataset,
       batch_size=25,
       shuffle=True
       )

       self.test_loader = DataLoader(
       dataset=self.test_dataset,
       batch_size=25
       )

    def train(self, num_epochs):
        train_model(self.model, self.train_loader, self.loss_fn,self.optimizer, num_epochs)
        torch.save(self.model.state_dict(), "binary_model.pt")
