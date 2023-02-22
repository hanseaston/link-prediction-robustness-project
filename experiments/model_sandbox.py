# NOTE: This code lets us run Python modules from different paths ###
import sys
import os
sys.path.append(os.getcwd())
#####################################################################


from models.GraphSAGE import GraphSAGE
from models.RandomWalk import Node2Vec

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import pandas as pd
import networkx as nx
import time

from dataset.utils import load_data, perturb_data

"""
This is an example of the methods we'll call on each model
"""

def main():
    start = time.time()

    train_test()
    load_test()
    # load_perturb_test()

    end = time.time()
    print(f"Script took {(end - start) / 60} minutes to run")


def train_test():
    """
    Trains and saves your model using abstract class methods
    """
    print("=> Preparing dataset...")
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    split_edge = dataset.get_edge_split()
    df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
    G = nx.from_pandas_edgelist(df)


    #### Different models might require slightly different settings here

    # print("=> Initializing GraphSAGE model...")
    # model = GraphSAGE()
    # print("=> Training model")
    # model.train(graph=G, val_edges=split_edge["valid"], epochs=1000)
    # print("=> Saving model...")
    # model.save_model()

    print("=> Initializing Node2Vec model...")
    model = Node2Vec()
    print("=> Training model")
    model.train(graph=G)
    print("=> Saving model...")
    model.save_model()

    ####


def load_test():
    """
    Loads and tests your model using abstract class methods
    """
    print("=> Preparing dataset...")
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    split_edge = dataset.get_edge_split()
    df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
    G = nx.from_pandas_edgelist(df)

    #### Different models might require slightly different settings here

    # print("=> Initializing GraphSAGE model...")
    # model = GraphSAGE()
    # print("=> Loading model")
    # model.load_model(model_path="models/trained_model_files/_gnn_dict_ep330.pt")

    print("=> Initializing GraphSAGE model...")
    model = Node2Vec()
    print("=> Loading model")
    model.load_model()

    ####
    
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

    # Validation set results:
    # GNN achieves: {'Hits@20': 0.6120878873914705, 'Hits@50': 0.6636501884050371, 'Hits@100': 0.6919970933934632}


def load_perturb_test():
    """
    Loads the data from dataset.utils.load_data() and tests the models performance on a
    perturbed dataset.
    """

    graph, split_dict = load_data()
    
    # NOTE: Pretty slow...
    #       - let's create a bunch of these and save them all
    proportions = [0.01, 0.1, 0.25]



    graph_dict = {}
    for prop in proportions:
        print(f"\tPerturbing data by {(prop * 100):2f}%")
        print("\t Before:\t", graph.number_of_edges())
        graph_dict[prop] = perturb_data(graph, proportion=prop, edges=split_dict)
        print("\t After:\t", graph.number_of_edges())

    # TODO: Train methods on these graphs...

    





if __name__ == "__main__":
    load_perturb_test()

