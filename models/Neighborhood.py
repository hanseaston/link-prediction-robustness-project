from models.LinkPredModel import LinkPredictor
import networkx as nx


class CommonNeighbors(LinkPredictor):

    def __init__(self) -> None:
        super().__init__()

    def train(self, graph: nx.Graph, **kwargs:dict) -> None:
        nodes = graph.number_of_nodes()
        print("how many nodes", nodes)
    
        for i in range(nodes):
            # need the community information for the algorithm to work, in this case, set it all to 0
            graph.nodes[i]["community"]=  0
        predictions = nx.cn_soundarajan_hopcroft(G=graph,ebunch=None)
        self.predictions = {}
        for u, v, p in predictions:
            self.predictions[(u, v)] = p


    def score_edge(self, node1:int, node2:int) -> float:
        return self.predictions[(node1, node2)]
    
    def score_edges(self, edge_list:list):
        scores = []
        for edge in edge_list:
            scores.append(self.predictions(edge.node1, edge.node2))

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