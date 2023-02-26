from models.LinkPredModel import LinkPredictor
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk, EdgeSplitter
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

import os
import pickle


class NodeToVec(LinkPredictor):
    def __init__(self):
        def node2vec_embedding(graph, name, num_walks=10, walk_length=80, p=1.0, q=1.0, dimensions=128, window_size=10, workers=4):
            rw = BiasedRandomWalk(graph)
            walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
            print(f"Number of random walks for '{name}': {len(walks)}")
            model = Word2Vec(
                walks,
                vector_size=dimensions,
                window=window_size,
                min_count=0,
                sg=1,
                workers=workers,
            )
            return model

        def link_prediction_classifier(max_iter=2000):
            lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
            return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

        def link_examples_to_features(link_examples, embedding, binary_operator):
            return [
                binary_operator(embedding.wv[src], embedding.wv[dst])
                for src, dst in link_examples
            ]

        def operator_hadamard(u, v):
            return u * v

        self.embed_fn = node2vec_embedding
        self.embedding = None # set during training
        self.link_to_features = link_examples_to_features
        self.link_pred_clf = link_prediction_classifier()
        self.bin_operator = operator_hadamard

    def train(self, graph, **kwargs):
        print("=> creating train splits ...")
        graph = StellarGraph.from_networkx(graph)
        edge_splitter = EdgeSplitter(graph)
        graph_train, examples_train, labels_train = edge_splitter.train_test_split(
            p=0.01, method="global"
        )
        print("=> creating train node embeddings ...")
        self.embedding = self.embed_fn(graph_train, "Train Graph")
        link_features = self.link_to_features(
            examples_train, self.embedding, self.bin_operator
        )
        print("=> training link prediction clf ...")
        self.link_pred_clf.fit(link_features, labels_train)
        return self.link_pred_clf

    def score_edge(self, node1, node2):
        edge_list = [[node1, node2]]
        pred_list = self.score_edges(edge_list, batch_size=1)
        return pred_list[0]
    

    def score_edges(self, edge_list, batch_size=-1):
        edge_features = self.link_to_features(edge_list, self.embedding, self.bin_operator)
        return self.link_pred_clf.predict(edge_features)


    def save_model(self, model_path=""):
        if len(model_path) == 0:
            model_path = "models/trained_model_files"
        pickle.dump(self.link_pred_clf, open(os.path.join(model_path, "clf.sav"), 'wb'))
        self.embedding.save(os.path.join(model_path, "embedding.model"))

    def load_model(self, model_path=""):
        if len(model_path) == 0:
            model_path = "models/trained_model_files"
        self.link_pred_clf = pickle.load(open(os.path.join(model_path, "clf.sav"), 'rb'))
        self.embedding = Word2Vec.load(os.path.join(model_path, "embedding.model"))