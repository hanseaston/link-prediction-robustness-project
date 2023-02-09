# NOTE: This code lets us run Python modules from different paths ###
import sys
import os
sys.path.append(os.getcwd())
#####################################################################

import pandas as pd
import networkx as nx
import gzip
import shutil
from ogb.linkproppred import PygLinkPropPredDataset
from models.Neighborhood import CommonNeighbors

# sets up the dataset and unzips the raw data (only needs to be run once)
def setup():
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    for file_path in ["dataset/ogbl_ddi/raw/edge.csv", \
                    "dataset/ogbl_ddi/raw/num-edge-list.csv", \
                    "dataset/ogbl_ddi/raw/num-node-list.csv"]:
        with gzip.open(file_path + ".gz", 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def main():
    # setup()
    df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
    G = nx.from_pandas_edgelist(df)
    model = CommonNeighbors()
    model.train(graph=G)
    score = model.score_edge(23, 1981)
    print(score)


if __name__ == "__main__":
    main()

