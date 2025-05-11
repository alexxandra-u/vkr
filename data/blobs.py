import torch
import numpy as np
import porespy as ps
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset

class CustomTensorDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.data = tensors[0]
        self.targets = tensors[1]

def generate_blobs():
    n_classes, samples_per_class, samples_per_class_test = 2, 6000, 1000
    params = {
        "0": {"porosity": 0.1, "blobiness": 1},
        "1": {"porosity": 0.12, "blobiness": 1.3}
    }

    X_train_blobs, y_train_blobs = [], []
    X_test_blobs, y_test_blobs = [], []

    for class_id in range(n_classes):
        porosity = params[f"{class_id}"]["porosity"]
        blobiness = params[f"{class_id}"]["blobiness"]

        for _ in range(samples_per_class):
            img = ps.generators.blobs(shape=[28, 28],
                                    porosity=porosity,
                                    blobiness=blobiness,
                                    seed=np.random.randint(0, 1000))
            X_train_blobs.append(img)
            y_train_blobs.append(class_id)

        for _ in range(samples_per_class_test):
            img = ps.generators.blobs(shape=[28, 28],
                                    porosity=porosity,
                                    blobiness=blobiness,
                                    seed=np.random.randint(0, 1000))
            X_test_blobs.append(img)
            y_test_blobs.append(class_id)

    X_train_np = np.array(X_train_blobs)
    y_train_np = np.array(y_train_blobs)
    X_test_np = np.array(X_test_blobs)
    y_test_np = np.array(y_test_blobs)

    X_train_np, y_train_np = shuffle(X_train_np, y_train_np, random_state=42)
    X_test_np, y_test_np = shuffle(X_test_np, y_test_np, random_state=42)

    X_train = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1) 
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(1) 
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    blobs_train = CustomTensorDataset(X_train, y_train)
    blobs_test = CustomTensorDataset(X_test, y_test)

    return blobs_train, blobs_test