import gzip
import shutil
from ogb.linkproppred import PygLinkPropPredDataset


def setup():
    # downloads the dataset if there's no local copy
    PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    # unzips all the raw data
    for file_path in ["dataset/ogbl_ddi/raw/edge.csv", \
                    "dataset/ogbl_ddi/raw/num-edge-list.csv", \
                    "dataset/ogbl_ddi/raw/num-node-list.csv"]:
        with gzip.open(file_path + ".gz", 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

# sets up the dataset on your local computer
if __name__ == "__main__":
    setup()