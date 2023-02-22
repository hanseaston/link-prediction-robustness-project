import networkx as nx
import random
from ogb.linkproppred import PygLinkPropPredDataset
import pandas as pd

def perturb_data(graph, edges, method="random", seed=123, proportion=0.1):
    """
    Given a networkx graph, returns new version of the graph with edges removed according to the
    given method. Proportion indicates the proportion of edges that are perturbed.

    NOTE: This method will mutate your original graph...?
    """

    random.seed(seed)
    graph_copy = graph.copy(as_view=False)


    match method:
        case "random_removal":
            return random_removal(graph_copy, edges, proportion)
        case "random_add":
            return random_add(graph_copy, edges, proportion)
        case "random_swap":
            return random_swap(graph_copy, edges, proportion)

        case _:
            raise Exception(f"{method} is not a supported method for edge removal.")


def random_removal(graph, edges, proportion):
    num_edges = graph.number_of_edges()
    num_removed = int(proportion * num_edges)
    idxs = random.sample(range(num_edges), num_removed)
    ebunch = [e for i, e in enumerate(graph.edges) if i in idxs]
    graph.remove_edges_from(ebunch)

def random_add(graph, edges, proportion):
    pass

def random_swap(graph, edges, proportion):
    pass


def load_data(test_as_tensor=False):
    """
    Loads the OGB-DDI dataset and returns a networkx graph for training, as well as the validation
    and test edges as a dicitonary e.g. "val" -> "pos" -> [(1, 2), (3, 4), ...]

    If test_as_tensor=True, returns the val/test split as tensor instead of edge list
    """
    
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    df = pd.read_csv("dataset/ogbl_ddi/raw/edge.csv", names=["source", "target"])
    G = nx.from_pandas_edgelist(df)

    split_edge = dataset.get_edge_split()

    if not test_as_tensor:
        split_dict = {
            "train": {
                "edge": split_edge["train"]["edge"].tolist(),
            },
            "valid": {
                "edge": split_edge["valid"]["edge"].tolist(),
                "edge_neg": split_edge["valid"]["edge_neg"].tolist()
            },
            "test": {
                "edge": split_edge["test"]["edge"].tolist(),
                "edge_neg": split_edge["test"]["edge_neg"].tolist()
            },
        }
    else:
        split_dict = {
            "train": {
                "edge": split_edge["train"]["edge"],
            },
            "valid": {
                "edge": split_edge["valid"]["edge"],
                "edge_neg": split_edge["valid"]["edge_neg"]
            },
            "test": {
                "edge": split_edge["test"]["edge"],
                "edge_neg": split_edge["test"]["edge_neg"]
            },
        }

    return G, split_dict

    
