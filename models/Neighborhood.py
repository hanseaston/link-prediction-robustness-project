from models.LinkPredModel import LinkPredictor
import networkx as nx


class JaccardSimilarity(LinkPredictor):

    def __init__(self) -> None:
        super().__init__()

    def train(self, graph: nx.Graph, **kwargs:dict) -> None:

        # calculating the jaccard similarity of all the nodes that do not have an existing edge
        jaccard_similarity = nx.jaccard_coefficient(graph, [(4039, 2424)])

        self.predictions = {}
        for u, v, p in jaccard_similarity:
            smaller = min(u, v)
            larger = max(u, v)
            self.predictions[(smaller, larger)] = p


    def score_edge(self, node1:int, node2:int) -> float:
        # ensures symmetric ordering
        smaller = min(node1, node2)
        larger = max(node1, node2)
        key = (smaller, larger)
        if key not in self.predictions:
            raise Exception("the nodes are found in graph")
        return self.predictions[key]
    
    def score_edges(self, edge_list:list):
        scores = []
        for edge in edge_list:
            scores.append(self.score_edge(edge))

    def save_model(self, model_path=None):
        pass
    
    def load_model(self, model_path=None):
        pass


class AdamicAdar(LinkPredictor):

    def __init__(self) -> None:
        super().__init__()

    def train(self, graph:list, **kwargs:dict) -> None:
        ...
    
    def score_edge(self, node1:int, node2:int) -> float:
        ...

    def save_model(self, model_path=None):
        ...
    
    def load_model(self, model_path=None):
        ...