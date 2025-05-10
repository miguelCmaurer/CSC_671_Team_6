import torch
from torch.utils.data import DataLoader, random_split
from severity_model.severity_mlp import MLP as Severity_MLP, train_model
from dataset.data_fetch import UCIDataset

data = UCIDataset()
data.severity_setup()

class Severity_model():
    def __init__(self):
        self.model = Severity_MLP(num_features = 13, dropout=.15)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.loss_fn = torch.nn.MSELoss()

        len_dataset = len(data)
        train_size = int(.8 * len_dataset)
        test_size = len_dataset - train_size

        self.train_dataset, self.test_dataset = random_split(data, [train_size, test_size])


        self.train_loader = DataLoader(
        dataset=self.train_dataset,
        batch_size=8,
        shuffle=True
        )

        self.test_loader = DataLoader(
        dataset=self.test_dataset,
        batch_size=16,
        )

    def train(self, num_epochs):
        train_model(self.model, self.train_loader, self.loss_fn, self.optimizer, num_epochs)
        torch.save(self.model.state_dict(), "severity_model.pt")
