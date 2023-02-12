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
        self.eigenvect = None
        self.eigenval = None
        self.adj_mat = None
        

    def train(self, graph:list, **kwargs:dict) -> None:
        adj_matrix = nx.adjacency_matrix(graph)
        eigenval, eigenvec = np.linalg.eig(adj_matrix.toarray())
        eigenval = np.real(eigenval)
        eigenvect = np.real(eigenvec)
        self.eigenvect = eigenvect
        self.eigenval = eigenval
        self.adj_mat = adj_matrix

    
    def score_edge(self, node1:int, node2:int) -> float:
        """
        calculating score using dotproduct of Q_r and P_T_i as shown in slide 20 of rec_sys2 lecture
        """
        # get the k = 100 projection 
        k = 500
        proj_matrix = self.eigenvect[:, :k]
        n1_proj = self.adj_mat[node1].T @ proj_matrix
        n2_proj = self.adj_mat[node2].T @ proj_matrix
        return np.dot(n1_proj, n2_proj)

    def score_edges(self, edge_list:list) -> np.ndarray:
        res = np.zeros(len(edge_list))
        for index, edge in enumerate(edge_list):
            node1, node2 = edge
            res[index] = self.score_edge(node1, node2)
        return res


    def save_model(self, model_path=None):
        import ipdb
        ipdb.set_trace()
        np.savez(model_path, self.eigenval, self.eigenvect, self.adj_mat.toarray())
    
    def load_model(self, model_path=None):
        npz = np.load(model_path)
        self.eigenval = npz['arr_0']
        self.eigenvect = npz['arr_1']
        self.adj_mat = npz['arr_2']
