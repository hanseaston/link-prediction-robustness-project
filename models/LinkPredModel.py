"""
Abstract class definition for our link prediction models.
"""

from abc import ABC, abstractmethod     # Abstract Base Class


class LinkPredictor(ABC):
    
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, graph, **kwargs:dict) -> None:
        """
        Given a networkx graph, initialize appropriate model parameters so that we can
        quickly determien whether there is an edge between two nodes. Optionally, can include
        other model specific parameters via kwargs.
        """
        pass

    @abstractmethod
    def score_edge(self, node1:int, node2:int) -> float:
        """
        Given two integers representing node indices, returns a score between indicating whether
        there should be an edge between the two nodes.
        """
        pass

    @abstractmethod
    def score_edges(self, edge_list:list) -> list:
        """
        Given a list of edges e.g., [[src1, target1], [src2, target2], ...], returns a numpy array
        of scores indicating whether that edge should be present.
        """
        pass

    @abstractmethod
    def save_model(self, model_path=None):
        """Save relevant model details to model_path"""
        pass

    @abstractmethod
    def load_model(self, model_path=None):
        """
        Loads the model from disk. This may require a default file path for where the
        model is stored.
        """
        pass