import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class GaussianSyntheticMixture(Dataset):
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.samples = []
        self.target = []
        
        self.generate_samples()

    def generate_samples(self):
        for _ in range(self.n):
            mus = np.array(np.meshgrid(*[(0, 4)] * self.k)).T.reshape(-1, self.k)
            cov = 1 / np.sqrt(self.k) * np.identity(self.k)
            which = np.random.choice(range(2 ** self.k))
            sample = np.random.multivariate_normal(mus[which], cov)
            self.samples.append(sample)
            self.target.append(which)
        self.samples = np.array(self.samples)

    def __getitem__(self, i):
        return torch.tensor(self.samples[i], dtype=torch.float32), self.target[i]
    
    def __len__(self):
        return self.n


class CustomMNIST(Dataset):
    def __init__(self, path, wanted_classes=range(10)):
        self.wanted_classes = list(wanted_classes)
        self.dataset = datasets.MNIST(path, train=True, download=True,
                                      transform=transforms.ToTensor())
        self._select_wanted_classes()

    def __getitem__(self, i):
        x, t = self.dataset[i]
        return x, t
    
    def _select_wanted_classes(self):
        self.dataset = [el for el in self.dataset if el[1] in self.wanted_classes]

    def __len__(self):
        return len(self.dataset)
    

class CustomCIFAR10(Dataset):
    def __init__(self, path, wanted_classes=range(10)):
        self.wanted_classes = list(wanted_classes)
        self.dataset = datasets.CIFAR10(path, train=True, download=True,
                                        transform=transforms.ToTensor())
        self._select_wanted_classes()

    def __getitem__(self, i):
        x, t = self.dataset[i]
        return x, t
    
    def _select_wanted_classes(self):
        self.dataset = [el for el in self.dataset if el[1] in self.wanted_classes]

    def __len__(self):
        return len(self.dataset)
    

class CustomFMNIST(Dataset):
    def __init__(self, path, wanted_classes=range(10)):
        self.wanted_classes = list(wanted_classes)
        self.dataset = datasets.FashionMNIST(path, train=True, download=True,
                                             transform=transforms.ToTensor())
        self._select_wanted_classes()

    def __getitem__(self, i):
        x, t = self.dataset[i]
        return x, t

    def _select_wanted_classes(self):
        self.dataset = [el for el in self.dataset if el[1] in self.wanted_classes]

    def __len__(self):
        return len(self.dataset)
    

class Brach3(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = pd.read_csv(self.path, delim_whitespace=True)
        self.scale()
        self.dataset = self.dataset.astype('float32')
        self.dataset.iloc[:, -1] = self.dataset.iloc[:, -1].astype('int')
        self.dataset.iloc[:, -1] -= 1

    def __getitem__(self, i):
        x = torch.tensor(self.dataset.iloc[i, :-1].values, dtype=torch.float32)
        t = self.dataset.iloc[i, -1]
        return x, t

    def __len__(self):
        return len(self.dataset)
    
    def scale(self):
        self.dataset.iloc[:, :-1] = (self.dataset.iloc[:, :-1] - self.dataset.iloc[:, :-1].mean()) / self.dataset.iloc[:, :-1].std()
        return self

class WineQuality(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = pd.read_csv(self.path, sep=';')
        self.scale()
        self.dataset = self.dataset.astype('float32')
        self.dataset.iloc[:, -1] = self.dataset.iloc[:, -1].astype('int')
        self.dataset.iloc[:, -1] -= self.dataset.iloc[:, -1].min()

    def __getitem__(self, i):
        x = torch.tensor(self.dataset.iloc[i, :-1].values, dtype=torch.float32)
        t = self.dataset.iloc[i, -1]
        return x, t
    
    def __len__(self):
        return len(self.dataset)
    
    def scale(self):
        self.dataset.iloc[:, :-1] = (self.dataset.iloc[:, :-1] - self.dataset.iloc[:, :-1].mean()) / self.dataset.iloc[:, :-1].std()
        return self
    

class Banknote(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = pd.read_csv(self.path)
        self.scale()
        self.dataset = self.dataset.astype('float32')
        self.dataset.iloc[:, -1] = self.dataset.iloc[:, -1].astype('int')

    def __getitem__(self, i):
        x = torch.tensor(self.dataset.iloc[i, :-1].values, dtype=torch.float32)
        t = self.dataset.iloc[i, -1]
        return x, t
    
    def __len__(self):
        return len(self.dataset)
    
    def scale(self):
        self.dataset.iloc[:, :-1] = (self.dataset.iloc[:, :-1] - self.dataset.iloc[:, :-1].mean()) / self.dataset.iloc[:, :-1].std()
        return self
