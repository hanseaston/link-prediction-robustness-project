"""
Abstract class definition for our link prediction models.
"""

from abc import ABC, abstractmethod     # Abstract Base Class

class model(ABC):
    
    def __init__(self) -> None:
        super().__init__()
