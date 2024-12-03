import torch
import torch.nn as nn


# 3 layers:
#   1. in layer (in_dim)
#   2. Tanh layer (linear_dim)
#   3. out layer (hidden_dim)
# for example MNIST dataset, where image 28x28 = 784

# use nnMDS(784, 20, 7840)
# dimension of in layer will be 784
# next layer will be Tahn, dimension 7840 (so from smaller to larger)
# last layer will be out with dimension 20 (large shrink)
# so short, 784 -> 7840 -> Tanh -> 20

class nnMDS(nn.Module):
    def __init__(self, in_dim : int, hidden_dim : int, linear_dim = None):
        super().__init__()
        if linear_dim is None:
            linear_dim = in_dim

        self.encode = nn.Sequential(
            nn.Linear(in_features = in_dim, out_features = linear_dim),
            nn.Tanh(),
            nn.Linear(in_features = linear_dim, out_features = hidden_dim)
            )

    def forward(X : torch.Tensor) -> torch.Tensor:
        return self.encode(X)

class nonmetric_nnMDS(nn.Module):
    def __init__(self, in_dim : int, out_dim : int, linear_dim = None):
        super().__init__()
        if linear_dim is None:
            linear_dim = in_dim

        self.nonmetric_encode = nn.Sequential(
            nn.Linear(in_features = in_dim, out_features = in_dim),
            nn.Sigmoid(),
        )

        self.encode = nn.Sequential(
            self.nonmetric_encode,
            nn.Linear(in_features = in_dim, out_features = linear_dim),
            nn.Tanh(),
            nn.Linear(in_features = linear_dim, out_features = out_dim)
            )

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean = 0.0, std = 1.0)
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(X : torch.Tensor) -> torch.Tensor:
        return self.encode(X)


# datasets for training

import h5py # for loading USPS data
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10

def get_USPS_data(path = './usps.h5', data_type = "train") -> tuple:
    with h5py.File(path, 'r') as hf:
        data = hf.get(data_type)
        x = data.get("data")[:]
        y = data.get("target")[:]
        y = y.reshape((y.shape[0], 1))
    return (x, y)

class USPSdataset(data.Dataset):
    __slots__ = ['use_gpu', 'x', 'y']
    
    def __init__(self, X = None, data_type = 'train', use_gpu = False):
        super().__init__()
        self.use_gpu = use_gpu

        if X is None:
            (self.x, self.y) = get_USPS_data(data_type)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self) -> int:
        return len(self.x)
    
# for getting dataloaders
def load_data(train = True, shuffle = True, batch_size = 256, num_workers = 8, path = './', use_gpu = False, database = 'USPS') -> data.DataLoader:

    if database == 'USPS':
        t_dataset = USPSdataset(path, data_type = ("train" if train else "test"), use_gpu = use_gpu)
        return data.DataLoader(t_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    elif database == 'MNIST': return data.DataLoader(MNIST(path, train = train, download = True, transform = transforms.ToTensor()), batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    elif database == 'SVHN': return data.DataLoader(SVHN(path, split = ("train" if train else "test"), download = True, transform = transforms.ToTensor()), batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    elif database == 'CIFAR10': return data.DataLoader(CIFAR10(path, train = train, download = True, transform = transforms.ToTensor()), batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    else : raise Exception(f'No dataset named {data}')

if __name__ == '__main__':
    train_data = CIFAR10('./', train = True, download = True)
    test_data = CIFAR10('./', train = False, download = True)
    print(train_data)
    print(test_data)
