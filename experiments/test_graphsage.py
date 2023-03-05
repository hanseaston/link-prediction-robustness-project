import sys
import os
sys.path.append(os.getcwd())
#####################################################################

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from models.GraphSAGE import GraphSAGE
import pandas as pd
import numpy as np
import networkx as nx
import torch
torch.set_warn_always(False)

from dataset.utils import load_data
import time

import matplotlib.pyplot as plt

TRAIN = False

"""
Sandbox for GraphSAGE link prediction
"""

def main():
    start = time.time()

    if TRAIN:
        train()
    else:
        load_test()

    # get_results()

    end = time.time()

    print(f"Script took {round((end - start) / 60, 2)} minutes to run")


def get_results():
    _, split_edge = load_data()

    
    # eps = [2, 3, 5, 6, 7, 9, 10, 13, 20, 21, 25, 29, 36, 38, 40, 46, 48, 51, 52, 63, 66, 67, \
    #            74, 75, 76, 78, 80, 88, 108, 112, 124]
    eps = []
    results_val = {
            'Hits@20': [],
            'Hits@50': [],
            'Hits@100': []
            }
    results_test = {
            'Hits@20': [],
            'Hits@50': [],
            'Hits@100': []
            }

    for ep in range(200):
        model_path = f"results/random/remove/0/gnn_trained/ep{ep}_gnn.pt"
        model = GraphSAGE()
        if not os.path.isfile(model_path):
            continue
        
        eps.append(ep)
        model.load_model(model_path)

        pos_valid_preds = model.score_edges(split_edge["valid"]["edge"])
        neg_valid_preds = model.score_edges(split_edge["valid"]["edge_neg"])

        pos_test_pred = model.score_edges(split_edge["test"]["edge"])
        neg_test_pred = model.score_edges(split_edge["test"]["edge_neg"])

        evaluator = Evaluator(name='ogbl-ddi')

        # metrics on validation test
        for K in [20, 50, 100]:
            evaluator.K = K
            hits = evaluator.eval({
                'y_pred_pos': np.array(pos_valid_preds),
                'y_pred_neg': np.array(neg_valid_preds),
            })[f'hits@{K}']

            results_val[f'Hits@{K}'].append(hits)

        
        # metrics on test test
        for K in [20, 50, 100]:
            evaluator.K = K
            hits = evaluator.eval({
                'y_pred_pos': np.array(pos_test_pred),
                'y_pred_neg': np.array(neg_test_pred),
            })[f'hits@{K}']

            results_test[f'Hits@{K}'].append(hits)
    

        print(ep)


    colormap = {
        20: "red",
        50: "green",
        100: "blue"
    }

    for K in [20, 50, 100]:
        plt.plot(eps, results_test[f'Hits@{K}'], label=f'Test Hits@{K}', color=colormap[K])

    for K in [20, 50, 100]:
        plt.plot(eps, results_val[f'Hits@{K}'], label=f'Val Hits@{K}', color=colormap[K], linestyle='--')

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Hits@K")
    plt.savefig("results/graphsage_training_hits.png")


def train():
    """
    Trains and saves model using abstract class methods
    """
    print("=> Preparing dataset...")
    G, split_edge = load_data(test_as_tensor=True)

    print("=> Initializing model...")
    model = GraphSAGE()

    print("=> Training model...")
    model.train(G, val_edges=split_edge["val"])

    print("=> Saving model...")
    model.save_model()



def load_test():
    """
    Loads and tests model using abstract class methods
    """
    print("=> Preparing dataset...")
    _, split_edge = load_data()

    print("=> Initializing model...")
    model = GraphSAGE()

    print("=> Loading model...")
    model_dir = "results/random/remove/0/gnn_trained/ep147_gnn.pt"
    model.load_model(model_dir)
    
    print("=> Testing model")
    print("\tScoring edges...")

    pos_valid_preds = model.score_edges(split_edge["valid"]["edge"])
    neg_valid_preds = model.score_edges(split_edge["valid"]["edge_neg"])

    pos_test_pred = model.score_edges(split_edge["test"]["edge"])
    neg_test_pred = model.score_edges(split_edge["test"]["edge_neg"])

    evaluator = Evaluator(name='ogbl-ddi')
    results = {}

    print("\tRunning evaluator on val...")

    # metrics on validation test
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': np.array(pos_valid_preds),
            'y_pred_neg': np.array(neg_valid_preds),
        })[f'hits@{K}']

        results[f'Hits@{K}'] = hits

    print("On val set, model achieves:\n")
    print(str(results))
    
    with open(f"results/graphsage.txt", 'w') as f:
        f.write("On validation set, model achieves:\n")
        f.write(str(results) + "\n\n")

    print("\tRunning evaluator on test...")

    # metrics on test test
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': np.array(pos_test_pred),
            'y_pred_neg': np.array(neg_test_pred),
        })[f'hits@{K}']

        results[f'Hits@{K}'] = hits

    print("On test set, model achieves:\n")
    print(str(results))
    
    with open(f'results/graphsage.txt', 'a') as f:
        f.write("On test set, model achieves:\n")
        f.write(str(results))






if __name__ == "__main__":
    main()

