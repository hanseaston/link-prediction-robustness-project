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

# TODO: Can get edge info by looking at the split dict
#       Edges should be in the same order os those


"""This file contains functions for aggregating edge scores into plots, figures, and tables.
Assumes a directory structure given by score_edges.py .
"""


def main():
    plot_hits()


def analyze_ranking():
    model2prop_perf = {}

    props = [0, 0.01, 0.1, 0.25, 0.5, 1]

    # Get performance for different models
    for pert in ["add", "remove"]:
        for prop in props:
            for model in ["gnn", "runtime_cn"]:

                score_path = f"results/scores/random/{pert}/{prop}/{model}_scores.pth"

                if not os.path.isfile(score_path):
                    print(f"Skipping:\t{model}\t{pert}\t{prop}")
                    continue

                score_dict = pickle.load(open(score_path, "rb"))

                hits = ...
                print(f"{model}\t{pert}\t{prop}\t{hits}")

                if model not in model2prop_perf:
                    model2prop_perf[model] = []
                
                if pert == "remove":
                    prop *= -1
                model2prop_perf[model].append((prop, hits))


def plot_hits():
    """ Analyzes model defined at top of file. Assumes directory structure given by running
    get scores.
    """

    out_dir = "results/figures"

    # Hits@K
    K = 20
    evaluator = Evaluator(name='ogbl-ddi')
    evaluator.K = K

    model_cmap = {
        "gnn": "red",
        "runtime_cn": "blue"
    }
    model_name = {
        "gnn": "GraphSAGE",
        "runtime_cn": "Common Neighbors"
    }

    props = [0, 0.01, 0.1, 0.25, 0.5, 1]

    model2prop_perf = {}

    # Get performance for different models
    for pert in ["add", "remove"]:
        for prop in props:
            for model in ["gnn", "runtime_cn"]:

                score_path = f"results/scores/random/{pert}/{prop}/{model}_scores.pth"

                if not os.path.isfile(score_path):
                    print(f"Skipping:\t{model}\t{pert}\t{prop}")
                    continue

                score_dict = pickle.load(open(score_path, "rb"))

                hits = evaluator.eval(score_dict)[f'hits@{K}']
                print(f"{model}\t{pert}\t{prop}\t{hits}")

                if model not in model2prop_perf:
                    model2prop_perf[model] = []
                
                if pert == "remove":
                    prop *= -1
                model2prop_perf[model].append((prop, hits))
                

    
    for model, color in model_cmap.items():
        data_list = model2prop_perf[model]
        data_list.sort(key = lambda pair: pair[0])
        x_val = [pair[0] for pair in data_list]
        y_val = [pair[1] for pair in data_list]
        plt.plot(x_val, y_val, label=model_name[model], c=color)

    # TODO: Maybe we could add dotted lines for unperturbed perfromance for each model
    plt.xlabel("Proportion Perturbed")
    # plt.xticks([-0.5, -0.25, -0.1, -0.01, 0, 0.01, 0.1, 0.25, 0.5, 1])
    plt.ylabel(f"Hits@{K}")
    plt.legend()
    plt.savefig(out_dir + f"/hits@{K}.png")



if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Script took {round((end - start) / 60, 2)} minutes to run")
