from models.LinkPredModel import LinkPredictor
import networkx as nx
import numpy as np


class MatrixFactorization(LinkPredictor):

    """
    As described by Link Prediction for Social Network Paper

    Get low rank approxiamation of adjacency matrix (rank k matrix M_k) -> noise reduction 
     - William idea - get similarity metrix of eigenvectors 
     - paper idea: use rank k matrix for Katz measure, common neighbors, 
    """

    def __init__(self) -> None:
        super().__init__()
        self.Q = None
        self.P_T = None
        

    def train(self, graph:list, **kwargs:dict) -> None:
        adj_matrix = nx.adjacency_matrix(graph)
        eigenval, eigenvec = np.linalg.eig(adj_matrix.toarray())
        eigenval = np.real(eigenval)
        eigenvect = np.real(eigenvec)
        self.Q = eigenval
        self.P_T = eigenvect

    
    def score_edge(self, node1:int, node2:int) -> float:
        """
        calculating score using dotproduct of Q_r and P_T_i as shown in slide 20 of rec_sys2 lecture
        """
        return np.dot(self.Q[node1], self.P_T[node2])

    def score_edges(self, edge_list:list) -> np.ndarray:
        res = np.zeros(len(edge_list))
        for index, edge in enumerate(edge_list):
            node1, node2 = edge
            res[index] = self.score_edge(node1, node2)
        return res


    def save_model(self, model_path=None):
        np.savez(model_path, self.Q, self.P_T)
    
    def load_model(self, model_path=None):
        npz = np.load(model_path)
        self.Q = npz['arr_0']
        self.P_T = npz['arr_0']
