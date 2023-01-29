from ogb.linkproppred import LinkPropPredDataset

dataset = LinkPropPredDataset(name = "ogbl-ddi", root = 'dataset/')

split_edges = dataset.get_edge_split()

training_dataset = split_edges["train"]["edge"]
validation_dataset = split_edges["valid"]["edge"]
testing_dataet = split_edges["test"]["edge"]

# Do whatever you want! 