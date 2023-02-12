from models.LinkPredModel import LinkPredictor
import networkx as nx
import pickle


class JaccardSimilarity(LinkPredictor):

    def __init__(self) -> None:
        super().__init__()

    def train(self, graph: nx.Graph, **kwargs:dict) -> None:

        # calculating the jaccard similarity of all the nodes that do not have an existing edge
        jaccard_similarity = nx.jaccard_coefficient(graph)

        self.predictions = {}
        for u, v, p in jaccard_similarity:
            smaller = min(u, v)
            larger = max(u, v)
            self.predictions[(smaller, larger)] = p

    def score_edge(self, node1:int, node2:int) -> float:
        # ensures consistent ordering
        smaller = min(node1, node2)
        larger = max(node1, node2)
        key = (smaller, larger)
        if key not in self.predictions:
            raise Exception("the nodes are found in graph")
        return self.predictions[key]
    
    def score_edges(self, edge_list:list):
        scores = []
        for node1, node2 in edge_list:
            scores.append(self.score_edge(node1, node2))
        return scores

    def save_model(self, model_path=None):
        with open("pickle/jaccard.pickle", 'wb') as handle:
            pickle.dump(self.predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self, model_path=None):
        with open("pickle/jaccard.pickle", 'rb') as handle:
            self.predictions = pickle.load(handle)


class AdamicAdar(LinkPredictor):

    def __init__(self) -> None:
        super().__init__()

    def train(self, graph:list, **kwargs:dict) -> None:
        # calculating the jaccard similarity of all the nodes that do not have an existing edge
        adamic_adar_index = nx.adamic_adar_index(graph)

        self.predictions = {}
        for u, v, p in adamic_adar_index:
            smaller = min(u, v)
            larger = max(u, v)
            self.predictions[(smaller, larger)] = p
    
    def score_edge(self, node1:int, node2:int) -> float:
        # ensures consistent ordering
        smaller = min(node1, node2)
        larger = max(node1, node2)
        key = (smaller, larger)
        if key not in self.predictions:
            raise Exception("the nodes are found in graph")
        return self.predictions[key]
    
    def score_edges(self, edge_list:list):
        scores = []
        for node1, node2 in edge_list:
            scores.append(self.score_edge(node1, node2))
        return scores

    def save_model(self, model_path=None):
        with open("pickle/adamicadar.pickle", 'wb') as handle:
            pickle.dump(self.predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self, model_path=None):
        with open("pickle/adamicadar.pickle", 'rb') as handle:
            self.predictions = pickle.load(handle)