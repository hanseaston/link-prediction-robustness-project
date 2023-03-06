# NOTE: This code lets us run Python modules from different paths ###
import sys
import os
sys.path.append(os.getcwd())
#####################################################################


from models.GraphSAGE import GraphSAGE
# from models.RandomWalk import Node2Vec
from models.node_to_vec import NodeToVec

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from stellargraph.data import EdgeSplitter
import pandas as pd
import networkx as nx
import numpy as np
import time

from dataset.utils import load_data, perturb_data


def get_graph():
    # dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    # split_edge = dataset.get_edge_split()
    df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
    G = nx.from_pandas_edgelist(df)
    return G


def get_edge_splits():
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    split_edge = dataset.get_edge_split()
    return {
            "valid": {
                "edge": split_edge["valid"]["edge"].tolist(),
                "edge_neg": split_edge["valid"]["edge_neg"].tolist()
            },
            "test": {
                "edge": split_edge["test"]["edge"].tolist(),
                "edge_neg": split_edge["test"]["edge_neg"].tolist()
            },
        }


def get_perturbed_graphs(ps=[1.0]):
    graph = get_graph()
    return [perturb_data(graph=graph, proportion=p) for p in ps]


def split_into_positive_and_negative_examples(examples, labels):
    pos = list(zip(examples, labels))
    neg = list(zip(examples, labels))
    pos = list(filter(lambda x: x[1] == 1, pos))
    neg = list(filter(lambda x: x[1] == 0, neg))
    pos = list(map(lambda x: x[0], pos))
    neg = list(map(lambda x: x[0], neg))
    return pos, neg #np.array(pos), np.array(neg)

def run_expt(full_graph, train_graph, model_path=""):
    p = 1e-2 # 0.01
    model = NodeToVec()
    
    # Build splits
    # edge_splitter_test = EdgeSplitter(graph)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    # graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
    #     p=p, method="global"
    # )
    edge_splitter_train = EdgeSplitter(train_graph)
    _, examples_train, labels_train = edge_splitter_train.train_test_split(
        p=p, method="global"
    )

    # Training    
    model.train(graph=full_graph, links=examples_train, labels=labels_train)
    
    # Saving and loading
    model.save_model(model_path)
    # model.load_model()

    # Testing
    # model.update_emb_for_testing(graph=graph_test)

    # Evaluation
    evaluator = Evaluator(name='ogbl-ddi')
    results = {}

    # positive_examples, negative_examples = split_into_positive_and_negative_examples(
    #     examples_test, labels_test,
    # )
    split_edge = get_edge_splits()
    positive_examples = split_edge["valid"]["edge"]
    negative_examples = split_edge["valid"]["edge_neg"]

    positive_preds = model.score_edges(positive_examples)
    negative_preds = model.score_edges(negative_examples)

    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': positive_preds,
            'y_pred_neg': negative_preds,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = hits
    print("\t Model achieves:")
    print("\t", results)


def main():
    proportions = [0.01, 0.1, 0.25]
    full_graph = get_graph()
    train_graphs = get_perturbed_graphs(proportions) #get_graph()
    for train_graph, prop in zip(train_graphs, proportions):
        print("=>"*80 )
        print("=> perturbation proportion:", prop)
        print("")
        # TODO remove
        # graph = graph.subgraph(list(range(300)))
        print("# nodes:", train_graph.number_of_nodes())
        print('# edges:', train_graph.number_of_edges())
        run_expt(full_graph, train_graph, model_path="model__node_to_vec__"+str(prop))


if __name__=="__main__":
    main()