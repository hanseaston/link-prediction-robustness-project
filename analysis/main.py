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
    props = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1]
    pert_type = "random"
    models = [
        "gnn",
        "runtime_cn",
        "trained_mf"
    ]
    out_dir = f"results/figures/{pert_type}"

    # plot_hits(props, pert_type, models, out_dir, K=20)

    rank_dict, baseline = get_rankings(props, pert_type, models)
    plot_average_edge_change(rank_dict, baseline, props, out_dir)


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



def plot_average_edge_change(rank_dict, baseline, props, out_dir):

    model_perf = {model: [] for model in baseline.keys()}

    for prop in props:
        for model in baseline.keys():
            
            baseline_rank = baseline[model]
            pert_rank = rank_dict[model][prop]

            percent_change = np.divide(pert_rank, baseline_rank)

            avg = np.mean(percent_change)
            std = np.std(percent_change)

            if model == "trained_mf":
                print(prop, avg)

            model_perf[model].append((avg, std))

    for model in baseline.keys():
        avg = [avg for avg, std in model_perf[model]]
        # err = [std for avg, std in model_perf[model]]
        plt.plot(props, avg, label=model_name[model], c=model_cmap[model])
        # plt.errorbar(props, avg, yerr=err, c=model_cmap[model], capsize=5, alpha=0.5)

    plt.plot([-0.5, 1], [1, 1], label="No Change", alpha=0.3, c="#e7298a", linestyle="--")
    plt.legend()
    plt.xlabel("Perturbation Proportion")
    plt.yscale("log")
    plt.ylabel("Average Change in Rank")

    plt.savefig(f"{out_dir}/avg_rank_change.png")



def plot_hits(props, pert_type, models, out_dir, K=20):
    """ Analyzes model defined at top of file. Assumes directory structure given by running
    get scores.
    """
    evaluator = Evaluator(name='ogbl-ddi')
    evaluator.K = K

    model2prop_perf = {}

    # Get performance for different models
    for pert in ["add", "remove"]:
        for prop in props:
            for model in models:

                score_path = f"results/scores/{pert_type}/{pert}/{prop}/{model}_scores.pth"
                
                if not os.path.isfile(score_path):
                    # NOTE: This is needed if some of the scores are still zipped
                    if os.path.isfile(score_path + ".gz"):
                        with gzip.open(score_path + ".gz", 'rb') as f_in:
                            with open(score_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    else:
                        print(f"\t=> Skipping:\t{model}\t{pert}\t{prop}")
                        # print(f"\t\t{score_path}.gz")
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
                

    
    for model in models:
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
