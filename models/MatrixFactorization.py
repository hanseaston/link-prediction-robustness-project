from models.LinkPredModel import LinkPredictor
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

class MLPLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLPLinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class MatrixFactorization(LinkPredictor): 
    
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.emb = None
        self.edge_index = None

    def train(self, graph:list, val_edges):
        num_nodes = graph.number_of_nodes()
        hidden_channels = 256
        num_layers = 3
        dropout = 0.5
        epochs = 200
        lr = 0.01
        batch_size = 64 * 1024

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb = torch.nn.Embedding(num_nodes, hidden_channels).to(device)

        self.predictor = MLPLinkPredictor(hidden_channels, hidden_channels, 1,
                              num_layers, dropout).to(device)

        self.predictor.reset_parameters()
        self.emb.reset_parameters()

        optimizer = torch.optim.Adam(
            list(self.emb.parameters()) + list(self.predictor.parameters()), lr=lr)

        pos_list = []
        edge_list = [[], []]
        seen_nodes = set()

        for node, nbr_dict in graph.adjacency():
            seen_nodes.add(node)
            for n in nbr_dict.keys():
                if n not in seen_nodes:
                    pos_list.append([node, n])
                edge_list[0].append(int(node))
                edge_list[0].append(int(n))
                edge_list[1].append(int(n))
                edge_list[1].append(int(node))

        pos_train_edge = torch.tensor(pos_list).to(device)
        edge_index = torch.tensor(edge_list).to(device)
        self.edge_index= edge_index

        if val_edges is not None:
            pos_val_edges = val_edges["edge"]
            neg_val_edges = val_edges["edge_neg"]

        min_loss = None

        for epoch in range(1, epochs + 1):
            self.predictor.train()
            total_loss = total_examples = 0

            for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                        shuffle=True):


                optimizer.zero_grad()

                edge = pos_train_edge[perm].t()

                pos_out = self.predictor(self.emb.weight[edge[0]], self.emb.weight[edge[1]])
                pos_loss = -torch.log(pos_out + 1e-15).mean()

                edge = negative_sampling(edge_index, num_nodes=self.emb.weight.size(0),
                                        num_neg_samples=perm.size(0), method='dense')

                neg_out = self.predictor(self.emb.weight[edge[0]], self.emb.weight[edge[1]])
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

                loss = pos_loss + neg_loss
                loss.backward()
                optimizer.step()

                num_examples = pos_out.size(0)
                total_loss += loss.item() * num_examples
                total_examples += num_examples

            edge = val_edges['edge'].t()

            pos_out = self.predictor(self.emb.weight[edge[0]], self.emb.weight[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            edge = negative_sampling(self.edge_index, num_nodes=self.emb.weight.size(0),
                                    num_neg_samples=perm.size(0), method='dense')

            neg_out = self.predictor(self.emb.weight[edge[0]], self.emb.weight[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            val_loss = pos_loss + neg_loss

            if (min_loss is None) or (val_loss < min_loss):
                print("saving")
                min_loss = val_loss
                self.save_model()

            print(epoch, min_loss)
                
        return total_loss / total_examples

    def score_edge(self, node1:int, node2:int) -> float:
        import ipdb
        ipdb.set_trace()
        self.predictor.eval()
        edge_list = [[node1, node2]]
        pred_list = self.score_edges(edge_list, batch_size=1)
        return pred_list[0]


    def score_edges(self, edge_list:list):
        batch_size = 64 * 1024
        self.predictor.eval()
        edges = torch.tensor(edge_list)

        preds = []
        for perm in DataLoader(range(edges.size(0)), batch_size):
            edge = edges[perm].t()
            preds += [self.predictor(self.emb.weight[edge[0]], self.emb.weight[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        pred_list = pred.detach().cpu().numpy() 
        return pred_list

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = "models/trained_model_files/mf_model.pt"
        torch.save({
            "emb": self.emb,
            "link_predictor": self.predictor,
            "edge_index": self.edge_index
        }, model_path)

    def load_model(self, model_path = None):
        if model_path is None:
            model_path = "models/trained_model_files/mf_model.pt"
        model_dict = torch.load(model_path)
        self.predictor = model_dict['link_predictor']
        self.emb = model_dict['emb']
        self.edge_index = model_dict["edge_index"]
