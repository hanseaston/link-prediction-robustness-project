import sys
import os
sys.path.append(os.getcwd())
#####################################################################

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from models.Neighborhood import JaccardSimilarity
import pandas as pd
import networkx as nx
import time

"""
Sandbox for Jaccard algorithm link prediction
"""

def main():
    start = time.time()

    # train_test()
    load_test()

    end = time.time()
    print(f"Script took {(end - start) / 60} minutes to run")


def train_test():
    """
    Trains and saves your model using abstract class methods
    """
    print("=> Preparing dataset...")
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    # TODO figure out whether that is the correct dataset to load
    df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
    G = nx.from_pandas_edgelist(df)

    print("=> Initializing Jaccard model...")
    model = JaccardSimilarity()

    print("=> Training model")
    model.train(G)

    print("=> Saving model...")
    model.save_model()



def load_test():
    """
    Loads and tests your model using abstract class methods
    """
    print("=> Preparing dataset...")
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    split_edge = dataset.get_edge_split()
    df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
    G = nx.from_pandas_edgelist(df)

    print("=> Initializing Jaccard model...")
    model = JaccardSimilarity()

    print("=> Loading model")
    model.load_model(model_path="models/trained_model_files/_gnn_dict_ep330.pt") # TODO: change this
    
    print("=> Testing model")
    pos_preds = model.score_edges(split_edge["valid"]["edge"].tolist())
    neg_preds = model.score_edges(split_edge["valid"]["edge_neg"].tolist())

    evaluator = Evaluator(name='ogbl-ddi')
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_preds,
            'y_pred_neg': neg_preds,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = hits
    print("\t Model achieves:")
    print("\t", results)


if __name__ == "__main__":
    main()

