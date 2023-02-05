"""
Abstract class definition for our link prediction models.
"""

from abc import ABC, abstractmethod     # Abstract Base Class

class model(ABC):
    
    def __init__(self) -> None:
        super().__init__()


    def train(graph) -> None:
        """
        Given a graph as a list of edges, initialize appropriate model parameters so that we can
        quickly determien whether there is an edge between two nodes 
        """

    def score(node1, node2) -> float:
        """
        Return a score indicating whether there should be an edge between the two nodes
        """