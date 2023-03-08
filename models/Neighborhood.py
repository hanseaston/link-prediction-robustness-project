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

class CommonNeighbor(LinkPredictor):

    def __init__(self) -> None:
        super().__init__()

    def train(self, graph: nx.Graph, **kwargs:dict) -> None:

        # For now, we are not considering the communities in the graph
        # To get around the community restriction in nx.cn_soundarajan_hopcroft,
        # we treat each node as a different community
        for i in range(graph.number_of_nodes()):
            graph.nodes[i]['community'] = i

        # calculating the common neighbor metrics of all the nodes that do not have an existing edge
        common_neighbors = nx.cn_soundarajan_hopcroft(graph)

        self.predictions = {}
        for u, v, p in common_neighbors:
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
        if model_path is None:
            model_path = "pickle/commonneighbor.pickle"
        else:
            model_path += "commonneighbor.pickle"
        with open(model_path, 'wb') as handle:
            pickle.dump(self.predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = "pickle/commonneighbor.pickle"
        else:
            model_path += "commonneighbor.pickle"
        
        with open(model_path, 'rb') as handle:
            self.predictions = pickle.load(handle)


class AdamicAdar(LinkPredictor):

    def __init__(self) -> None:
        super().__init__()

    def train(self, graph:list, **kwargs:dict) -> None:
        # calculating the adamic-adar index of all the nodes that do not have an existing edge
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
        if model_path is None:
            model_path = "pickle/adamicadar.pickle"

        with open(model_path, 'wb') as handle:
            pickle.dump(self.predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = "pickle/adamicadar.pickle"
        
        with open(model_path, 'rb') as handle:
            self.predictions = pickle.load(handle)


class RuntimeCN(LinkPredictor):
    """
    Stores graph as list of [0,1] vectors.
    Hamming distance (i.e. dot prod) between vectors gives number of common neighbors.
    Call it Runtime Common Neighbors because it computes predictions on the fly rather than
    saving them in the model.
    --> Testing to see if this is faster than the other methods
    """

    def __init__(self) -> None:
        super().__init__()

    def train(self, graph, **kwargs:dict) -> None:
        self.adj_mat = nx.to_scipy_sparse_array(graph, nodelist=range(graph.number_of_nodes()), format="csr")

        # TODO: Testing that all degrees line up
        for n in graph:
            deg = len(list(graph[n]))
            sparse_deg = self.adj_mat.getrow(n).getnnz()
            if deg != sparse_deg:
                print(n, deg, sparse_deg)

        print(self.adj_mat.shape)
    
    def score_edge(self, node1:int, node2:int) -> float:
        return self.adj_mat.getrow(node1).dot(self.adj_mat.getrow(node2).T)
    
    def score_edges(self, edge_list:list):
        scores = []
        for node1, node2 in edge_list:
            scores.append(self.score_edge(node1, node2))
        return scores

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = "pickle/runtime_cn.pickle"
        elif model_path[-7:] != ".pickle":
            model_path += "/runtime_cn.pickle"

        with open(model_path, 'wb') as handle:
            pickle.dump(self.adj_mat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = "pickle/runtime_cn.pickle"
        elif model_path[-7:] != ".pickle":
            model_path += "/runtime_cn.pickle"
        
        with open(model_path, 'rb') as handle:
            self.adj_mat = pickle.load(handle)