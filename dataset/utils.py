import networkx as nx
from ogb.linkproppred import PygLinkPropPredDataset
import pandas as pd
import csv
import torch


def load_data(test_as_tensor=False, perturbation_path=None):
    """
    Loads the OGB-DDI dataset and returns a networkx graph for training, as well as the validation
    and test edges as a dicitonary e.g. "val" -> "pos" -> [(1, 2), (3, 4), ...]

    If test_as_tensor=True, returns the val/test split as tensor instead of edge list
    """
    if perturbation_path is None:
        dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
        df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
        G = nx.from_pandas_edgelist(df)
    else:
        dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
        df = pd.read_csv(perturbation_path, names=["source", "target"])
        G = nx.from_pandas_edgelist(df)

    split_edge = dataset.get_edge_split()

    if perturbation_path is None:
        if not test_as_tensor:
            split_dict = {
                "train": {
                    "edge": split_edge["train"]["edge"].tolist(),
                },
                "valid": {
                    "edge": split_edge["valid"]["edge"].tolist(),
                    "edge_neg": split_edge["valid"]["edge_neg"].tolist()
                },
                "test": {
                    "edge": split_edge["test"]["edge"].tolist(),
                    "edge_neg": split_edge["test"]["edge_neg"].tolist()
                },
            }
        else:
            split_dict = {
                "train": {
                    "edge": split_edge["train"]["edge"],
                },
                "valid": {
                    "edge": split_edge["valid"]["edge"],
                    "edge_neg": split_edge["valid"]["edge_neg"]
                },
                "test": {
                    "edge": split_edge["test"]["edge"],
                    "edge_neg": split_edge["test"]["edge_neg"]
                },
            }
        return G, split_dict
    else:
        train_edges = []
        with open(perturbation_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                train_edges.append([int(row[0]), int(row[1])])
        csv_file.close()

        if not test_as_tensor:
            split_dict = {
                "train": {
                    "edge": train_edges,
                },
                "valid": {
                    "edge": split_edge["valid"]["edge"].tolist(),
                    "edge_neg": split_edge["valid"]["edge_neg"].tolist()
                },
                "test": {
                    "edge": split_edge["test"]["edge"].tolist(),
                    "edge_neg": split_edge["test"]["edge_neg"].tolist()
                },
            }
        else:
            split_dict = {
                "train": {
                    "edge": torch.Tensor(train_edges), # TODO: not too sure about this, need to double check
                },
                "valid": {
                    "edge": split_edge["valid"]["edge"],
                    "edge_neg": split_edge["valid"]["edge_neg"]
                },
                "test": {
                    "edge": split_edge["test"]["edge"],
                    "edge_neg": split_edge["test"]["edge_neg"]
                },
            }
        return G, split_dict
    
        
