import sys
import os
import gzip
# !pip install torch_geometric
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool,GATConv
from torch_geometric.nn import ChebConv
from torch_geometric.data import Dataset
# from torch_geometric.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch.optim.lr_scheduler import StepLR,MultiStepLR

modelpath=sys.argv[2]
traindatasetpath=sys.argv[4]
validdatasetpath=sys.argv[6]

trainpath=traindatasetpath
validpath=validdatasetpath

import torch
import gzip
from torch_geometric.data import Data, Dataset, DataLoader, Batch

class MyDataset:
    def __init__(self, path):
        # Load gzipped data
        self.f1 = path + '/node_features.csv.gz'
        self.f2 = path + '/num_nodes.csv.gz'
        self.f3 = path + '/num_edges.csv.gz'
        self.f4 = path + '/edges.csv.gz'
        self.f5 = path + '/graph_labels.csv.gz'
        self.f6 = path + '/edge_features.csv.gz'
        self.graph_label = []
        self.data_list = []
        self.edge = []
        self.node_features = []
        self.edge_features = []
        self.node_feat = []
        self.edge_feat = []
        self.edges = [[], []]
        self.data_list = []
        self.graph_label = []
        self.process_csv(self.f1, self.f2, self.f3, self.f4, self.f5)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def process_csv(self, f1, f2, f3, f4, f5):
        with gzip.open(self.f5, 'rt') as file:
            for line in file:
                # line = line[:-1]
                if 'nan' not in line:
                    self.graph_label.append([(float(line))])
                else:
                    self.graph_label.append([int(1)])

        self.graph_label = torch.tensor(self.graph_label)

        with gzip.open(self.f4, 'rt') as file:
            for line in file:
                self.edges[0].append(list(map(int, line.split(',')))[0])
                self.edges[1].append(list(map(int, line.split(',')))[1])

        with gzip.open(self.f3, 'rt') as file:
            c = 0
            for k, line in enumerate(file):
                t = []
                line = line[:-1]
                t.append(self.edges[0][c:int(float(line)) + c])
                t.append(self.edges[1][c:int(float(line)) + c])
                self.edge.append(torch.tensor(t))
                c += int(float(line))

        with gzip.open(self.f1, 'rt') as file:
            for line in file:
                self.node_feat.append(list(map(float, line.split(','))))

        self.node_feat = torch.tensor(self.node_feat)

        with gzip.open(self.f6, 'rt') as file:
            for line in file:
                self.edge_feat.append(list(map(float, line.split(','))))

        self.edge_feat = torch.tensor(self.edge_feat)

        with gzip.open(self.f2, 'rt') as file:
            c = 0
            for line in file:
                self.node_features.append(self.node_feat[c:int(float(line)) + c])
                c += int(line)

        with gzip.open(self.f3, 'rt') as file:
            c = 0
            for line in file:
                self.edge_features.append(self.edge_feat[c:int(float(line)) + c])
                c += int(line)

        l = len(self.graph_label)
        for i in range(l):
            if len(self.edge[i][0]) > 0:
                data = Data(x=self.node_features[i], edge_index=self.edge[i], y=self.graph_label[i],
                            edge_attr=self.edge_features[i])
                self.data_list.append(data)

        return self.data_list


# Instantiate the dataset
# dataset = MyDataset(trainpath)

# Create a PyTorch DataLoader

train=MyDataset(trainpath)
validate=MyDataset(validpath)



def calculate_roc_auc(model, loader):
    model.eval()
    all_labels = []
    all_probs = []

    # Get the device information from the model
    device = next(model.parameters()).device

    with torch.no_grad():
        for data in loader:
            # Move data to the same device as the model
            data = data.to(device)

            output = model(data)
            probs = torch.exp(output)[:, 1]  # Probability for class 1
            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Check if both classes are present in the validation set
    if len(np.unique(all_labels)) == 1:
        print("Warning: Only one class present in the validation set. ROC AUC score is not defined.")
        return None

    return roc_auc_score(all_labels, all_probs)

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob, num_heads=1):
        super(GATModel, self).__init__()

        self.node_encoder = nn.Sequential(
            GATConv(input_dim, hidden_dim, heads=num_heads),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_prob),
            GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.edge_attr

        # Manually apply GAT layers in sequence
        for layer in self.node_encoder:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        x = global_add_pool(x, data.batch)

        x = self.mlp(x)

        return x

import matplotlib.pyplot as plt

def plot_metrics_and_save(roc_auc_scores, fold_val_losses, fold_train_losses, save_path=None):
    epochs = range(1, len(roc_auc_scores) + 1)

    # Plot ROC-AUC scores
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, roc_auc_scores, label='ROC-AUC Score', marker='o')
    plt.title('ROC-AUC Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC Score')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path + 'roc_auc_plot.png')
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fold_val_losses, label='Validation Loss', marker='o', color='orange')
    plt.title('Validation Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path + 'validation_loss_plot.png')
    plt.show()

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fold_train_losses, label='Training Loss', marker='o', color='green')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path + 'training_loss_plot.png')
    plt.show()

# Example usage:
# plot_metrics_and_save(roc_auc_scores, fold_val_losses, fold_train_losses, save_path='/your/save/directory/')

learning_rate=0.0015
num_epochs=30
best_roc_auc = 0.0
input_dim=train[0].x.shape[1]
hidden_dim=9
output_dim=2
dropout_prob=0.2
num_heads=1
best_roc_auc = 0.0
batch_size=12

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

model = GATModel(input_dim, hidden_dim, output_dim, dropout_prob,num_heads)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.NLLLoss()

fold_train_losses = []
fold_val_losses = []
roc_auc_scores = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        # print(output,data.y.long())
        loss = criterion(output, data.y.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    fold_train_losses.append(train_loss / len(train_loader))
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in validate:
            output = model(data)
            val_loss += criterion(output, data.y.long()).item()
            # _, predicted = output.max(1)
    fold_val_losses.append(val_loss / len(validate))
    # Calculate ROC-AUC for validation set
    roc_auc = calculate_roc_auc(model, validate)
    roc_auc_scores.append(roc_auc)
    # print(roc_auc,train_loss)

plot_metrics_and_save(roc_auc_scores, fold_val_losses, fold_train_losses)

torch.save(model.state_dict(), modelpath)