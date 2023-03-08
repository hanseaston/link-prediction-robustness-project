import sys
import os
sys.path.append(os.getcwd())
#####################################################################

import gzip
import shutil
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

    props = [0, 0.1, 0.25, 0.5, 1]

    # TODO: Look at change in ranking for each positive edge

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
                    prop_print = prop * -1
                else:
                    prop_print = prop
                model2prop_perf[model].append((prop_print, hits))


def plot_hits():
    """ Analyzes model defined at top of file. Assumes directory structure given by running
    get scores.
    """

    # Hits@K
    K = 20
    evaluator = Evaluator(name='ogbl-ddi')
    evaluator.K = K

    pert_type = "random"

    model_name = {
        "gnn": "GraphSAGE",
        "runtime_cn": "Common Neighbors",
        "trained_mf": "Matrix Factorization"
    }

    props = [0, 0.1, 0.25, 0.5, 1]


    model2prop_perf = {}
    model_cmap = {
        "gnn": "red",
        "runtime_cn": "blue",
        "trained_mf": "green"
    }
    out_dir = f"results/figures/{pert_type}"

    # Get performance for different models
    for pert in ["add", "remove"]:
        for prop in props:
            for model in model_name.keys():

                score_path = f"results/scores/{pert_type}/{pert}/{prop}/{model}_scores.pth"
                
                if not os.path.isfile(score_path):
                    # NOTE: This is needed if some of the scores are still zipped
                    if os.path.isfile(score_path + ".gz"):
                        with gzip.open(score_path + ".gz", 'rb') as f_in:
                            with open(score_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    else:
                        print(f"=> Skipping:\t{model}\t{pert}\t{prop}")
                        print(f"\t\t{score_path}.gz")
                        continue

                score_dict = pickle.load(open(score_path, "rb"))

                start = time.time()
                hits = evaluator.eval(score_dict)[f'hits@{K}']
                end = time.time()

                print(f"{model}\t{pert}\t{prop}\t{hits}\t time: {end - start}")
                for k, v in score_dict.items():
                    print("\t", model, type(v[0]), v[0])
                

                if model not in model2prop_perf:
                    model2prop_perf[model] = []
                
                if pert == "remove":
                    prop_print = prop * -1
                else:
                    prop_print = prop

                model2prop_perf[model].append((prop_print, hits))
                

    
    for model in model_name.keys():
        color = model_cmap[model]
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
