import networkx as nx
from node2vec import Node2Vec as n2v
from models.LinkPredModel import LinkPredictor
from gensim.models import KeyedVectors
from numpy.linalg import norm


class Node2Vec(LinkPredictor):
    """
    LinkPredModel wrapper around implementation of node2vec from this repo
    https://github.com/eliorc/node2vec
    """

    def __init__(self) -> None:
        super().__init__()

        self.embeddings = None

    def train(self, graph, dimensions=64, walk_length=30, num_walks=200, workers=4) -> None:
        # NOTE: Not deterministic if workers != 1
        node2vec = n2v(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)

        # This is a genism model
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Keyed dictionary of embeddings as numpy vectors
        self.embeddings = model.wv

    
    def score_edge(self, node1:int, node2:int) -> float:
        a = self.embeddings[str(node1)]
        b = self.embeddings[str(node2)]
        eps = 1e-15         # Small constant for stability
        return (a @ b.T) / (norm(a)*norm(b) + eps)

    def score_edges(self, edges:list) -> list:
        return [self.score_edge(e[0], e[1]) for e in edges]

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = "models/trained_model_files/node2vec.wordvectors"
        
        # Store just the nodes + their trained embeddings.
        node_vectors = self.embeddings
        node_vectors.save(model_path)

    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = "models/trained_model_files/node2vec.wordvectors"

        # Load back with memory-mapping = read-only, shared across processes.
        self.embeddings = KeyedVectors.load(model_path, mmap='r')