import networkx as nx
from ogb.linkproppred import PygLinkPropPredDataset
import pandas as pd
import csv
import torch

NUM_NODES = 4267

def load_data(test_as_tensor=False, perturbation_path=None):
    """
    Loads the OGB-DDI dataset and returns a networkx graph for training, as well as the validation
    and test edges as a dicitonary e.g. "val" -> "pos" -> [(1, 2), (3, 4), ...]

    If test_as_tensor=True, returns the val/test split as tensor instead of edge list
    """
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')

    if perturbation_path is None:
        df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
        G = nx.from_pandas_edgelist(df)
    else:
        df = pd.read_csv(perturbation_path, names=["source", "target"])
        G = nx.from_pandas_edgelist(df)

    # TODO: This is not a great final solution. When applying link prediction to graphs, we
    #       don't consider nodes with no neighbors at all
    # Add back all nodes that were disconnected during perturbation
    for node in range(NUM_NODES):
        if not G.has_node(node):
            print(node, "not in graph")
            G.add_node(node)
        

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
                # print(f"`{row}`")
                if len(row) == 0:   # skip blank lines
                    continue
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
    
        
