

import os
import sys
import pickle
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

import argparse

class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob, num_heads):
        super(RegressionModel, self).__init__()

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
            # nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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

def tocsv(y_arr, *, task):

    import os
    import numpy as np
    import pandas as pd
    assert task in ["classification", "regression"], f"task must be either \"classification\" or \"regression\". Found: {task}"
    assert isinstance(y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze().shape) == 1, f"y_arr must be a vector. shape found: {y_arr.shape}"
    assert not os.path.isfile(f"y_{task}.csv"), f"File already exists. Ensure you are not calling this function multiple times (e.g. when looping over batches). Read the docstring. Found: y_{task}.csv"
    y_arr = y_arr.squeeze()
    df = pd.DataFrame(y_arr)
    df.to_csv(f"y_{task}.csv", index=False, header=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluating the classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    print(f"Evaluating the classification model. Model will be loaded from {args.model_path}. Test dataset will be loaded from {args.dataset_path}.")


if __name__=="__main__":
    main()

path=sys.argv[4]

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

validate=MyDataset(path)
def custom_collate(batch):
        return Batch.from_data_list(batch)

validation_data = DataLoader(validate, batch_size=12, shuffle=True, collate_fn=custom_collate)

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_add_pool
# from train import RegressionModel
input_dim=0
for i,j in enumerate(validation_data):
    if(i==1):
        input_dim=j.x.shape[1]


model_path = sys.argv[2]
learning_rate=0.05
dout=0.2
batch_size=10
learning_rate=0.01
num_epochs=5
best_roc_auc = 0.0
best_model = None
num_folds=2
# input_dim=validation_data[0].x.shape[1]
hidden_dim=9
output_dim=1
dropout_prob=0.2
num_heads=1
model = RegressionModel(input_dim, hidden_dim, output_dim, dropout_prob, num_heads)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.5)
criterion = nn.MSELoss()

model.load_state_dict(torch.load(model_path))

# print(model)

ally=[]
model.eval()
with torch.no_grad():
  for data in validation_data:
      outputs = model(data)
      bat_list=outputs.cpu().numpy().tolist()
      ally.extend(bat_list)
np_ys=np.asarray(ally)
tocsv(np_ys,task="regression")