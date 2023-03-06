import sys
import os
sys.path.append(os.getcwd())
#####################################################################

## TODO: Write a script for analyzing

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from models.GraphSAGE import GraphSAGE
import pandas as pd
import numpy as np
import networkx as nx
import torch
import pickle
torch.set_warn_always(False)

from dataset.utils import load_data
import time

import matplotlib.pyplot as plt


"""...
"""

def main():
    start = time.time()

    get_scores()

    # analyze_models()

    end = time.time()

    print(f"Script took {round((end - start) / 60, 2)} minutes to run")

def save_scores(model, output_path):
    """
    Saves model scores to an an output path. Should end in .pth.
    """
    _, split_edge = load_data()
    
    print("=> Testing model")
    print("\tScoring edges...")

    pos_test_pred = model.score_edges(split_edge["test"]["edge"])
    neg_test_pred = model.score_edges(split_edge["test"]["edge_neg"])

    score_dict = {
        'y_pred_pos': np.array(pos_test_pred),
        'y_pred_neg': np.array(neg_test_pred),
    }

    pickle.dump(score_dict, open(output_path, "wb"))


def get_scores():

    # TODO: Define what model type you're using here
    model_type = GraphSAGE
    model_name = "gnn"

    # TODO: Create a list of model paths and the corresponding output path to use
    # NOTE: You don't need to use list comprehension here, just whatever works!
    model_list = [
        f"results/random/{pert}/{prop}/{model_name}.pt"
            for pert in ["add", "remove"]
                for prop in [0, 0.01, 0.1, 0.25, 0.5, 1]
    ]

    # NOTE: Please put your models scores in this directory tho: `results/scores/random/{pert}/{prop}`
    #       where `pert`` and `prop`` are the perturbation types (e.g., add, remove) and prop is the
    #       proportion of the dataset that was perturbed.
    output_paths = [
        f"results/scores/random/{pert}/{prop}/{model_name}_scores.pth"
            for pert in ["add", "remove"]
                for prop in [0, 0.01, 0.1, 0.25, 0.5, 1]
    ]

    for model_path, output_path in zip(model_list, output_paths):
        if not os.path.isfile(model_path):  # Skip model paths that haven't been trained
            print(f"\tSkipping {model_path}...")
            continue
        model = model_type()
        model.load_model(model_path)
        # Make all directories in the path
        output_dir = os.path.join(*output_path.split("/")[:-1])
        os.makedirs(output_dir, exist_ok=True)
        save_scores(model, output_path)


def analyze_models():

    # TODO: For all versions of a model
    # - What is the relative change in performance as a function of perturbation
    # - 


    eps = []

    _, split_edge = load_data()

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

    for ep in eps:
        model_path = f"results/random/add/0.25/gnn_training/ep{ep}_gnn.pt"
        model = GraphSAGE()
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











if __name__ == "__main__":
    main()

