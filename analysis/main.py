import sys
import os
sys.path.append(os.getcwd())
#####################################################################

import gzip
import shutil
from ogb.linkproppred import Evaluator

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
    props = [-0.5, -0.25, -0.1, -0.01, 0, 0.01, 0.1, 0.25, 0.5, 1]
    pert_type = "random"
    models = [
        "gnn",
        "runtime_cn",
        "trained_mf"
    ]
    out_dir = f"results/figures/{pert_type}"
    
    # for K in [20, 50, 100]:
    #     plot_hits(props, pert_type, models, out_dir, K=K)

    rank_dict, baseline = get_rankings(props, pert_type, models)

    # plot_average_edge_change(rank_dict, baseline, props, out_dir)

    error_analysis(rank_dict, baseline, 0.5, out_dir)


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


def error_analysis(rank_dict, baseline, props, out_dir): 

    model = "gnn"
    baseline_rank = baseline[model]

    G, split_dict = load_data()
    pos_test_edges = split_dict["test"]["edge"]
    avg_degs = [(G.degree[node1] + G.degree[node2])/2 for node1, node2 in pos_test_edges]


    prop_cmap = {
        0.1: "#d95f02",
        0.25: "#7570b3",
        0.5: "#1b9e77"
    }

    props = [
        0.5,
        # 0.25,
        0.1
    ]

    for prop in props:
        pert_rank = rank_dict[model][prop]
        prop_change = np.divide(pert_rank, baseline_rank)
        plt.scatter(avg_degs, prop_change, alpha=0.2, c=prop_cmap[prop], label=prop)

    plt.plot([0, 1600], [1, 1], label="No Change", c="#8dd3c7", linestyle="--")
    plt.legend()
    plt.xlabel("Average Degree")
    plt.ylabel("Average Change in Rank")
    plt.yscale("log")

    plt.savefig(f"{out_dir}/degree_scatter.png")
    plt.clf()


def plot_average_edge_change(rank_dict, baseline, props, out_dir):

    model_perf = {model: [] for model in baseline.keys()}

    for prop in props:
        for model in baseline.keys():
            
            baseline_rank = baseline[model]
            pert_rank = rank_dict[model][prop]
            if pert_rank is None:
                continue
            
            # i.e., number of times worse pertrubed is than baseline on avg
            prop_change = np.divide(pert_rank, baseline_rank)

            avg = np.mean(prop_change)
            std = np.std(prop_change)

            if model == "trained_mf":
                print(prop, avg)

            model_perf[model].append(((avg, prop), std))

    for model in baseline.keys():
        xval = [prop for (avg, prop), std in model_perf[model]]
        avg = [avg for (avg, prop), std in model_perf[model]]
        err = [std for (avg, prop), std in model_perf[model]]
        plt.plot(xval, avg, label=model_name[model], c=model_cmap[model])
        # plt.errorbar(xval, avg, yerr=err, c=model_cmap[model], capsize=5, alpha=0.5)

    plt.plot([props[0], props[-1]], [1, 1], label="No Change", alpha=0.3, c="#e7298a", linestyle="--")
    plt.legend()
    plt.xlabel("Perturbation Proportion")
    plt.yscale("log")
    plt.ylabel("Average Change in Rank")

    plt.savefig(f"{out_dir}/avg_rank_change.png")
    plt.clf()



def plot_hits(props, pert_type, models, out_dir, K=20):
    """ Analyzes model defined at top of file. Assumes directory structure given by running
    get scores.
    """
    evaluator = Evaluator(name='ogbl-ddi')
    evaluator.K = K

    model2prop_perf = {}

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
                if os.path.isfile(score_path + ".gz"):
                    with gzip.open(score_path + ".gz", 'rb') as f_in:
                        with open(score_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    continue

            score_dict = pickle.load(open(score_path, "rb"))

            hits = evaluator.eval(score_dict)[f'hits@{K}']

            if model not in model2prop_perf:
                model2prop_perf[model] = []
            

            model2prop_perf[model].append(hits)
                

    
    for model in models:
        color = model_cmap[model]
        y_val = model2prop_perf[model]
        base = y_val[props.index(0)]

        # Just trying to see MF 0.01, change remove this and fix props soon
        if len(props) != len(y_val):
            plt.plot([-0.5, -0.25, -0.1, -0.01, 0, 0.1, 0.25, 0.5, 1], y_val, label=model_name[model], c=color)
            plt.plot([props[0], props[-1]], [base, base], c=color, linestyle="--", alpha=0.5)
            continue

        plt.plot(props, y_val, label=model_name[model], c=color)
        plt.plot([props[0], props[-1]], [base, base], c=color, linestyle="--", alpha=0.5)

    # TODO: Maybe we could add dotted lines for unperturbed perfromance for each model
    plt.xlabel("Proportion Perturbed")
    plt.ylabel(f"Hits@{K}")
    plt.legend()
    plt.savefig(out_dir + f"/hits@{K}.png")
    plt.clf()



if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Script took {round((end - start) / 60, 2)} minutes to run")
