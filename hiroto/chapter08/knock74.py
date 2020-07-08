import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
from knock73 import Net

with open("./data/train_labels.pickle", mode="rb") as train_l,\
        open("./data/valid_labels.pickle", mode="rb") as valid_l,\
        open("./data/test_labels.pickle", mode="rb") as test_l,\
        open("./data/train_vectors.pickle", mode="rb") as train_v,\
        open("./data/valid_vectors.pickle", mode="rb") as valid_v,\
        open("./data/test_vectors.pickle", mode="rb") as test_v,\
        open("./models/73model.pickle", mode="rb") as model_file:
    train_labels = pickle.load(train_l)
    valid_labels = pickle.load(valid_l)
    test_labels = pickle.load(test_l)
    train_vectors = pickle.load(train_v)
    valid_vectors = pickle.load(valid_v)
    test_vectors = pickle.load(test_v)
    model = pickle.load(model_file)


def select_class(probs_list):
    labels = []
    for probs in probs_list:
        label = np.argmax(probs.detach().numpy())
        labels.append(label)
    return labels


def main():
    outputs = model(train_vectors)
    pred = select_class(outputs)
    true = train_labels.numpy()
    print(f"train accuracy: {accuracy_score(true, pred)}")
    outputs = model(valid_vectors)
    pred = select_class(outputs)
    true = valid_labels.numpy()
    print(f"valid accuracy: {accuracy_score(true, pred)}")


if __name__ == "__main__":
    main()

'''
train accuracy: 0.935698240359416
valid accuracy: 0.9019461077844312
'''