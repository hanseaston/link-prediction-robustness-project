import sys
import os
sys.path.append(os.getcwd())
#####################################################################


import gzip
import shutil
from ogb.linkproppred import Evaluator

from itertools import compress

import pandas as pd
import numpy as np
import networkx as nx
import torch
import pickle
torch.set_warn_always(False)

from dataset.utils import load_data
import time

import matplotlib.pyplot as plt


"""This file contains functions for aggregating edge scores into plots, figures, and tables.
Assumes a directory structure given by score_edges.py .
"""

# Plotting maps

# NOTE: Use color brewer for all colors (https://colorbrewer2.org/#type=qualitative&scheme=Set3&n=4)
model_cmap = {
        "gnn": "#1b9e77",
        "runtime_cn": "#d95f02",
        "trained_mf": "#7570b3"
    }

model_name = {
        "gnn": "GraphSAGE",
        "runtime_cn": "Common Neighbors",
        "trained_mf": "Matrix Factorization"
    }



def main():
    print("Running analysis...")
    props = [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1]
    pert_type = "adversial"
    models = [
        "gnn",
        "runtime_cn",
        "trained_mf"
    ]
    out_dir = f"results/figures/{pert_type}"
    
    # print("Plotting hits...")
    # for K in [20, 50, 100]:
    #     plot_hits(props, pert_type, models, out_dir, K=K)

    print("Ranking edges...")
    rank_dict, baseline = get_rankings(props, pert_type, models)

    error_analysis_degree(rank_dict, baseline, out_dir, pert_type)

    # print("Plotting average edge change...")
    # plot_average_edge_change(rank_dict, baseline, props, out_dir)

    # error_analysis(rank_dict, baseline, 0.5, out_dir)


def get_rankings(props, pert_type, models):
    """
    Returns a dictionary containing the positive edge rankings for each perturbed dataset.
    Also returns the rankings on the baseline dataset.
    """

    # Dictionary mapping model type to dictionary of perturbation proportions to list of
    # positive edge ranks E.g. rank_dict["gnn"]["0.5"] <- [1, 3, 97, 3, ....]
    rank_dict = {
        "gnn": {},
        "runtime_cn": {},
        "trained_mf":{}
    }
    baseline = {}

    # Get performance for different models
    for prop in props:
        if prop <= 0:
            pert = "remove"
            print_prop = -1 * prop
        else:
            pert = "add"
            print_prop = prop

        for model in models:
            score_path = f"results/scores/{pert_type}/{pert}/{print_prop}/{model}_scores.pth"
            if not os.path.isfile(score_path):
                rank_dict[model][prop] = None
                continue

            score_dict = pickle.load(open(score_path, "rb"))
            y_pred_pos, y_pred_neg = score_dict['y_pred_pos'], score_dict['y_pred_neg']

            pos_rank = len(y_pred_neg) - np.searchsorted(np.sort(y_pred_neg), y_pred_pos, side="right") + 1
            rank_dict[model][prop] = pos_rank
    
    # Get baseline model performance
    for model in models:
        score_path = f"results/scores/random/remove/0/{model}_scores.pth"
        score_dict = pickle.load(open(score_path, "rb"))
        y_pred_pos, y_pred_neg = score_dict['y_pred_pos'], score_dict['y_pred_neg']
        pos_rank = len(y_pred_neg) - np.searchsorted(np.sort(y_pred_neg), y_pred_pos, side="right") + 1
        baseline[model] = pos_rank


    return rank_dict, baseline


def error_analysis_degree(rank_dict, baseline, out_dir, pert_type): 

    model = "gnn"
    baseline_rank = baseline[model]

    G_original, split_dict = load_data()
    pos_test_edges = split_dict["test"]["edge"]
    avg_degs_original = [(G_original.degree[node1] + G_original.degree[node2])/2 for node1, node2 in pos_test_edges]



    prop_cmap = {
        0.1: "#d95f02",
        0.25: "#7570b3",
        0.5: "#1b9e77"
    }

    props = [
        0.5,
        0.25,
        0.1
    ]

    for prop in props:
        pert_rank = rank_dict[model][prop * -1]

        # NOTE: I am using substract, rather than using porpotion
        prop_change = np.abs(np.subtract(pert_rank, baseline_rank))

        perturbed_path = f"dataset/perturbation/{pert_type}_remove_{prop}.csv"
        print("Perturbed path", perturbed_path)
        G_perturbed, _ = load_data(perturbation_path=perturbed_path)

        avg_degs_perturbed = [(G_perturbed.degree[node1] + G_perturbed.degree[node2])/2 for node1, node2 in pos_test_edges]
        
        change_in_degs = np.abs(np.array(avg_degs_perturbed) - np.array(avg_degs_original))


        plt.scatter(change_in_degs, prop_change, alpha=0.2, c=prop_cmap[prop], label=prop)

    plt.plot([0, 1600], [1, 1], label="No Change", c="#8dd3c7", linestyle="--")
    plt.legend()
    plt.xlabel("Average Change in Degree")
    plt.ylabel("Average Change in Rank")
    plt.yscale("log")

    plt.show()

    # plt.savefig(f"{out_dir}/degree_scatter.png")
    # plt.clf()



if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Script took {round((end - start) / 60, 2)} minutes to run")
