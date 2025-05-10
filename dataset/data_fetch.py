import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.impute import KNNImputer

class UCIDataset(Dataset):
    def __init__(self, filePath='dataset/heart_disease.csv', binary=False):
        self.filePath = filePath
        data = np.genfromtxt(filePath, delimiter=',', skip_header=0, dtype=float, missing_values="?", filling_values=np.nan)
        imputer = KNNImputer(n_neighbors=5)
        data = imputer.fit_transform(data)
        self.x = torch.tensor(data[:, :-1], dtype=torch.float32)
        self.y = torch.tensor(data[:, -1], dtype=torch.float32)
        if binary:
            self.y = (self.y > 0).float()
        self.normalize()

    def normalize(self):
        self.x_mean = self.x.mean(dim=0, keepdim=True)
        self.x_std = self.x.std(dim=0, keepdim=True)
        self.normalized_x = (self.x - self.x_mean) / self.x_std

    def severity_setup(self):
        self.x = self.x[self.y != 0]
        self.y = self.y[self.y != 0]

    def __getitem__(self, index):
        return self.normalized_x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]
