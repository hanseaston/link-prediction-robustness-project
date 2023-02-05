from models import LinkPredModel


class MatrixFactorization(LinkPredModel):

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
