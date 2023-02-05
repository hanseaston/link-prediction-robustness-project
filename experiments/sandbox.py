import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import matplotlib.pyplot as plt
import numpy as np

"""
This file serves as a way of exploring the contents of OGB ddi dataset. We would like to extract
the training/test/val splits from the tensors loaded by OGB into a format that is more amenable to
our other methods (e.g., as a networkx graph).

...
"""


def main():
    # Download the dataset onto disk
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')

    # This generates a dictionary with train/test/val split of the edges we use

    split_edge = dataset.get_edge_split()

    # E.g., all edges used for training as a torch tensor w/ torch.Size([1067911, 2])
    training_set = split_edge['train']['edge']


    # Unzip raw data files
    import gzip
    import shutil

    for file_path in ["dataset/ogbl_ddi/raw/edge.csv", \
                      "dataset/ogbl_ddi/raw/num-edge-list.csv", \
                      "dataset/ogbl_ddi/raw/num-node-list.csv"]:
        with gzip.open(file_path + ".gz", 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # raw/edge.csv contains the graph as a list of edges that we will use

    mapping_dir = "dataset/ogbl_ddi/mapping/"
    for file in ["ddi_description", "nodeidx2drugid"]:
        with gzip.open(mapping_dir + file + ".csv.gz", 'rb') as f_in:
            with open(mapping_dir + file + ".csv", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # mapping/ddi_description.csv contains all edges in drug bank (about 2x as many as our dataset)


    import pandas as pd
    import networkx as nx

    # Let's try to read the raw edges into a pandas dataframe and load it as a graph into networkx

    print("Reading from csv...")
    df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
    print(df.head())

    print("Building networkx graph...")
    G = nx.from_pandas_edgelist(df)
    print(f"\tDDI graph has", G.number_of_edges(), "edges")
    print(f"\tDDI graph has", G.number_of_nodes(), "nodes")
    print(f"\tSparsity proportion is {1 - G.number_of_edges() / G.number_of_nodes()**2}")

    # Get some stats about the graph
    cc = nx.connected_components(G)
    biggest_cc = []
    for nodes in cc:
        if len(nodes) > len(biggest_cc):
            biggest_cc = G.subgraph(nodes)
    print(f"\tBiggest connected component", len(biggest_cc))    # Graph is composed of a single connected component (probably processed entire drugbank graph to get this)

    total_degree = 0
    for node in G.nodes():
        succ = [n for n in G.adj[node]]
        total_degree += len(succ)
    print(f"\tAverage degree is", total_degree / len(G))

    training_set = split_edge['train']['edge']  # torch.Size([1067911, 2])
    val_set = split_edge['valid']['edge']       # torch.Size([133489, 2])
    test_set = split_edge['test']['edge']       # torch.Size([133489, 2])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 12))
    nx.draw(G, alpha=0.3, node_size=200, width=0.5)
    plt.savefig("results/ddi_draw.png")
    plt.clf()

    plt.figure(figsize=(12, 12))
    nx.draw_spectral(G, alpha=0.3, node_size=200, width=0.5)
    plt.savefig("results/ddi_spectral.png")
    plt.clf()

    plt.figure(figsize=(12, 12))
    nx.draw_shell(G, alpha=0.3, node_size=200, width=0.5)
    plt.savefig("results/ddi_shell.png")
    plt.clf()





if __name__ == "__main__":
    main()