# NOTE: This code lets us run Python modules from different paths ###
import sys
import os
sys.path.append(os.getcwd())
#####################################################################

from ogb.linkproppred import PygLinkPropPredDataset
from models.GraphSAGE import GraphSAGE
import pandas as pd
import networkx as nx
import time

"""
This is an example of the methods we'll call on each model
"""

start = time.time()

print("=> Preparing dataset...")
dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
split_edge = dataset.get_edge_split()
df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
G = nx.from_pandas_edgelist(df)

print("=> Initializing GraphSAGE model...")
model = GraphSAGE()

print("=> Training model")
model.train(graph=G, val_edges=split_edge["valid"], epochs=1000)

print("=> Saving model...")
model.save_model()


end = time.time()
print(f"Script took {(end - start) / 60} minutes to run")

