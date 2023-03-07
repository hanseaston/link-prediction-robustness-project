import sys
import os
sys.path.append(os.getcwd())
#####################################################################

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from models.GraphSAGE import GraphSAGE
from models.Neighborhood import RuntimeCN
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
    model_type, model_name, model_list, output_paths = get_params("runtime_cn")

    # NOTE: This line only needs to run once
    get_scores(model_type, model_name, model_list, output_paths)

    # NOTE: Use this to check that you've properly saved edges + scores
    test_models(output_paths)



def test_models(score_paths):
    """ Analyzes model defined at top of file. Assumes directory structure given by running
    get scores.
    """

    for score_path in score_paths:
        if not os.path.isfile(score_path):  # Skip paths that haven't been trained
            continue
        
        results_test = {}
        
        # Load scores
        score_dict = pickle.load(open(score_path, "rb"))
        
        # metrics on test test
        evaluator = Evaluator(name='ogbl-ddi')
        for K in [20, 50, 100]:
            evaluator.K = K
            hits = evaluator.eval(score_dict)[f'hits@{K}']

            results_test[f'Hits@{K}'] = hits

        print(f"{score_path}\n\t{results_test}")


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


def get_scores(model_type, model_name, model_list, output_paths):

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


def get_params(model_name):
    """ Big switch statement to get the correct model paths for each model.
    """

    if model_name == "runtime_cn":
        # Define what model type you're using here
        model_type = RuntimeCN
        model_name = "runtime_cn"

        # Create a list of model paths and the corresponding output path to use
        # NOTE: You don't need to use list comprehension here, just whatever works!
        model_list = [
            f"results/random/{pert}/{prop}/{model_name}.pickle"
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

        # model_list = [
        #     f"results/random/{pert}/{prop}/{model_name}.pickle"
        #         for pert in ["add"]
        #             for prop in [1]
        # ]

        # output_paths = [
        #     f"results/scores/random/{pert}/{prop}/{model_name}_scores.pth"
        #         for pert in ["add"]
        #             for prop in [1]
        # ]
    
    elif model_name == "gnn":
        model_type = GraphSAGE
        model_name = "gnn"

        model_list = [
            f"results/random/{pert}/{prop}/{model_name}.pt"
                for pert in ["add", "remove"]
                    for prop in [0, 0.01, 0.1, 0.25, 0.5, 1]
        ]

        output_paths = [
            f"results/scores/random/{pert}/{prop}/{model_name}_scores.pth"
                for pert in ["add", "remove"]
                    for prop in [0, 0.01, 0.1, 0.25, 0.5, 1]
        ]


    # TODO: Add your models here
    # elif:
    #     ...

    else:
        raise Exception(f"Did not recognize {model_name}")

    return model_type, model_name, model_list, output_paths







if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Script took {round((end - start) / 60, 2)} minutes to run")
