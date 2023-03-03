import sys
import os
sys.path.append(os.getcwd())
#####################################################################

from models.GraphSAGE import GraphSAGE
from models.Neighborhood import CommonNeighbor, AdamicAdar, RuntimeCN
from models.RandomWalk import Node2Vec

from ogb.linkproppred import Evaluator

import pandas as pd
import numpy as np
import networkx as nx

from dataset.utils import load_data
import time


model_types = [
    (GraphSAGE, "graphsage"),
    # (CommonNeighbor, "commonneighbors"),
    # (AdamicAdar, "adamicadar"),
    # (RuntimeCN, "runtime_cn"),
    # (Node2Vec, "node2vec")
]

def main():

    perturb_dir = "dataset/perturbation"
    _, split_edge_tensor = load_data(test_as_tensor=True)
    _, split_edge_list = load_data()

    for perturb_type in ["random"]:
        # for change in ["remove", "add"]:
        #     for prop in [0.25, 0.1, 0.01]:
        for change in ["add"]:
            for prop in [0.25]:
            
                data_path = perturb_dir + f"/{perturb_type}_{change}_{prop}.csv"
                G, _ = load_data(perturbation_path=data_path)

                for LinkPredictor, name in model_types:
                    start = time.time()

                    out_path = f"results/{perturb_type}/{change}/{prop}/"
                    os.makedirs(out_path, exist_ok=True)
                    
                    print(f"==> Training {name}: {perturb_type} {change} {prop}")
                    model = LinkPredictor()

                    if name == "graphsage":
                        model.train(G, val_edges=split_edge_tensor["valid"], out_path=out_path)
                    else:
                        model.train(G)

                    model.save_model(out_path)

                    print(f"==> Testing")
                    model.load_model(out_path)

                    # TODO: Make train/val/test format independent of model type
                    if name == "graphsage":
                        split_edge = split_edge_tensor
                    else:
                        split_edge = split_edge_list
                    

                    pos_valid_preds = model.score_edges(split_edge["valid"]["edge"])
                    neg_valid_preds = model.score_edges(split_edge["valid"]["edge_neg"])

                    pos_test_pred = model.score_edges(split_edge["test"]["edge"])
                    neg_test_pred = model.score_edges(split_edge["test"]["edge_neg"])
                    print("\tEdges scored")

                    evaluator = Evaluator(name='ogbl-ddi')
                    results = {}

                    # metrics on validation test
                    for K in [20, 50, 100]:
                        evaluator.K = K
                        hits = evaluator.eval({
                            'y_pred_pos': np.array(pos_valid_preds),
                            'y_pred_neg': np.array(neg_valid_preds),
                        })[f'hits@{K}']

                        results[f'Hits@{K}'] = hits

                    print("\tVal scoring evaluated")
                    
                    with open(f"{out_path}/{name}.txt", 'w') as f:
                        f.write("On validation set, model achieves:\n")
                        f.write(str(results) + "\n\n")

                    # metrics on test test
                    for K in [20, 50, 100]:
                        evaluator.K = K
                        hits = evaluator.eval({
                            'y_pred_pos': np.array(pos_test_pred),
                            'y_pred_neg': np.array(neg_test_pred),
                        })[f'hits@{K}']

                        results[f'Hits@{K}'] = hits
                    
                    with open(f'{out_path}/{name}.txt', 'a') as f:
                        f.write("On test set, model achieves:\n")
                        f.write(str(results))

                    end = time.time()

                    print(f"\tScript took {round((end - start) / 60, 2)} minutes to run")





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
    # NOTE: The ep440 gnn does not have edge_index. Need to re-save the ep440 by
    # assigning it ep330 edge index
    model.load_model("models/trained_model_files/graphsage/_gnn_dict_ep330.pt")
    
    print("=> Testing model...")

    pos_valid_preds = model.score_edges(split_edge["valid"]["edge"])
    neg_valid_preds = model.score_edges(split_edge["valid"]["edge_neg"])

    pos_test_pred = model.score_edges(split_edge["test"]["edge"])
    neg_test_pred = model.score_edges(split_edge["test"]["edge_neg"])

    evaluator = Evaluator(name='ogbl-ddi')
    results = {}

    # metrics on validation test
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': np.array(pos_valid_preds),
            'y_pred_neg': np.array(neg_valid_preds),
        })[f'hits@{K}']

        results[f'Hits@{K}'] = hits
    
    with open(f"results/graphsage.txt", 'w') as f:
        f.write("On validation set, model achieves:\n")
        f.write(str(results) + "\n\n")

    # metrics on test test
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': np.array(pos_test_pred),
            'y_pred_neg': np.array(neg_test_pred),
        })[f'hits@{K}']

        results[f'Hits@{K}'] = hits
    
    with open(f'results/graphsage.txt', 'a') as f:
        f.write("On test set, model achieves:\n")
        f.write(str(results))



if __name__ == "__main__":
    main()
