
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

from ogb.linkproppred import Evaluator
import numpy as np
import os


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

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


def train(predictor, node_embs, pos_train_edge, edge_index, optimizer, batch_size):
    predictor.train()

    total_loss = total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()

        pos_out = predictor(node_embs[edge[0]], node_embs[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=node_embs.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(node_embs[edge[0]], node_embs[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

class RandomWalkMLP():
    def __init__(self) -> None:
        self.device = 0
        self.log_steps = 1
        self.num_layers = 3
        self.hidden_channels = 256
        self.dropout = 0.5
        self.batch_size = 64 * 1024
        self.lr = 0.01
        self.epochs = 200
        self.eval_steps = 5
        self.link_predictor = None
        self.embedding_path = None

    def save_model(self, model_path=None): 
        torch.save({
            "link_predictor": self.link_predictor,
            "node_embs": self.node_embs
        }, model_path)
    
    def load_model(self, model_path=None):        
        model_dict = torch.load(model_path)
        self.link_predictor = model_dict["link_predictor"]
        self.node_embs = model_dict["node_embs"]
    
    def score_edge(self, node1, node2):
        self.link_predictor.eval()
        edge_list = [[node1, node2]]
        pred_list = self.score_edges(edge_list, batch_size=1)
        return pred_list[0]
    

    def score_edges(self, edge_list, batch_size=1):
        self.link_predictor.eval()
        edges = torch.tensor(edge_list)
        preds = []
        for perm in DataLoader(range(edges.size(0)), batch_size):
            edge = edges[perm].t()
            x = self.link_predictor(self.node_embs[edge[0]], self.node_embs[edge[1]]).squeeze().cpu()
            preds.append(x.detach().numpy())
        return preds

    def train(self, graph, val_edges, embedding_path, out_path):   
        device = 'cpu'
        device = torch.device(device)

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

        self.node_embs = torch.load(embedding_path, map_location='cpu').to(device)

        evaluator = Evaluator(name='ogbl-ddi')
        
        self.link_predictor = LinkPredictor(self.node_embs.size(-1), self.hidden_channels, 1,
                                self.num_layers, self.dropout).to(device)

        optimizer = torch.optim.Adam(self.link_predictor.parameters(), lr=self.lr)

        max_val = -1
        for epoch in range(self.epochs):
            loss = train(predictor = self.link_predictor,
                         pos_train_edge = pos_train_edge,
                         node_embs = self.node_embs,
                         edge_index = edge_index,
                         optimizer = optimizer,
                         batch_size = self.batch_size)
            
            print(f"Epoch {epoch + 1}: loss: {round(loss, 5)}")

            result = {}

            pos_valid_preds = self.score_edges(val_edges["edge"])
            neg_valid_preds = self.score_edges(val_edges["edge_neg"])

            # metrics on validation test
            for K in [20]:
                evaluator.K = K
                hits = evaluator.eval({
                    'y_pred_pos': np.array(pos_valid_preds),
                    'y_pred_neg': np.array(neg_valid_preds),
                })[f'hits@{K}']

                result[f'Hits@{K}'] = hits

            val_performance = result['Hits@20']

            # print results every 10 iterations
            if (epoch + 1) % 10 == 0:
                print(result)

            # only save model file if the results increase in performance
            if val_performance > max_val:
                os.makedirs(f"{out_path}/randomwalk_trained/", exist_ok=True)
                self.save_model(model_path=f"{out_path}/randomwalk_trained/ep{epoch}_randomwalk.pt")
                max_val = val_performance
                print("=> Performance improvement for Hits@20 =", max_val)