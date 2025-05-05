import sys
sys.path.append("./src/")

import random
import numpy as np

import torch
import torch.nn as nn

import pickle
from torchvision.datasets import MNIST, KMNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from torcheval.metrics.functional import multiclass_accuracy, multiclass_auroc
# from torcheval import MulticlassAccuracy, MulticlassAUROC

from src.data import PersistenceTransformDataset, collate_fn

from src.models.phtx import PersistentHomologyTransformer
from src.trainer import fit

# model params
m = "PHTX"
d_model = 64
d_hidden = 192
num_heads = 8
num_layers = 4
dropout = 0

# optimization params
batch_size = 32
lr = 2.5e-4
epochs = 200

# dataset params
idx = [0, 2, 4, 6]
eps = 0.05

# randomness
seed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# random state
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# load data
dataset_base_train = MNIST(root="./data/_tmp/", train=True, download=True)
dataset_base_test = MNIST(root="./data/_tmp/", train=False, download=True)
dataset_diagrams_train = pickle.load(open("./data/MNIST_D_train_dir.pkl", "rb"))
dataset_diagrams_test = pickle.load(open("./data/MNIST_D_test_dir.pkl", "rb"))

# dataset and dataloader
dataset_train = PersistenceTransformDataset(dataset_base_train, dataset_diagrams_train, idx=torch.tensor(idx), eps=eps)
dataset_test = PersistenceTransformDataset(dataset_base_test, dataset_diagrams_test, idx=torch.tensor(idx), eps=eps)
dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, collate_fn=collate_fn) 
dataloader_test = DataLoader(dataset_test, batch_size, collate_fn=collate_fn)

# model
model = PersistentHomologyTransformer(None, 4, 10, d_model, d_hidden, num_heads, num_layers, dropout)
optimizer = Adam(model.parameters(), lr)
loss_fn = nn.CrossEntropyLoss()
metric_fn = multiclass_accuracy # accuracy, roc auc, f1

# fit, log
print("Data:\t\t idx=[{}], eps={}".format(", ".join(map(str, idx)), eps))
print("Model:\t\t d_model={}, d_hidden={}, num_heads={}, num_layers={}, dropout={}".format(d_model, d_hidden, num_heads, num_layers, dropout))
print("Optimization:\t lr={}, batch size={}, seed={}".format(lr, batch_size, seed))
_, history_model = fit(model, optimizer, loss_fn, metric_fn, epochs, dataloader_train, dataloader_test, desc=m)

np.save("./history_MNIST_full.npy", history_model)