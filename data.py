import colour
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from sklearn import datasets as sk_datasets

# points in 3d space
points_3d_train = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 3], [0, 3, 8], [10, 3, 15]]).float()
points_3d_test = torch.tensor([[0, 0, 2], [5, 0, 5]])
# train_dataset = TensorDataset(points_3d_train)
# test_dataset = TensorDataset(points_3d_test)

# munsell colour dataset
munsell_data = colour.notation.munsell.MUNSELL_COLOURS_ALL
lab_colours = torch.from_numpy(np.array([colour.convert(data[1], 'CIE XYZ', 'CIE Lab') for data in munsell_data])).float()
lab_colours_train = lab_colours[:int(len(lab_colours)*0.8)]
lab_colours_test = lab_colours[int(len(lab_colours)*0.8):]
train_dataset = TensorDataset(lab_colours_train)
test_dataset = TensorDataset(lab_colours_test)

# mnist
transform = transforms.ToTensor()
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
mnist_trainset = torch.stack([mnist_trainset[i][0] for i in range(len(mnist_trainset))])
mnist_testset = torch.stack([mnist_testset[i][0] for i in range(len(mnist_testset))])
# mnist_trainset = TensorDataset(mnist_trainset)
# mnist_testset = TensorDataset(mnist_testset)

# iris
iris_data = torch.from_numpy(sk_datasets.load_iris()['data']).float()
iris_trainset = iris_data[0:int(len(iris_data)*0.8)]
iris_testset = iris_data[int(len(iris_data)*0.8):]
# train_dataset = TensorDataset(iris_trainset)
# test_dataset = TensorDataset(iris_testset)

# wine
wine_data = torch.from_numpy(sk_datasets.load_wine()['data']).float()
wine_trainset = wine_data[0:int(len(wine_data)*0.8)]
wine_testset = wine_data[int(len(wine_data)*0.8):]
# train_dataset = TensorDataset(wine_trainset)
# test_dataset = TensorDataset(wine_testset)

# swiss roll
sr_data = torch.from_numpy(sk_datasets.make_swiss_roll()[0]).float()
sr_trainset = sr_data[0:int(len(sr_data)*0.8)]
sr_testset = sr_data[int(len(sr_data)*0.8):]
# train_dataset = TensorDataset(sr_trainset)
# test_dataset = TensorDataset(sr_testset)

