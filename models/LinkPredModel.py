"""
Abstract class definition for our link prediction models.
"""

from abc import ABC, abstractmethod     # Abstract Base Class

class model(ABC):
    
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, graph:list, **kwargs:dict) -> None:
        """
        Given a graph as a list of edges, initialize appropriate model parameters so that we can
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