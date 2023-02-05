# LinkPredictionOGB
Machine Learning for Big Data Project on Link Prediction


### Resources:
1. [GraphSAGE paper](https://arxiv.org/pdf/1706.02216.pdf)
2. [GNN survey paper](https://arxiv.org/pdf/1901.00596.pdf)
3. [Link Prediction method paper](http://www.eecs.harvard.edu/~michaelm/CS222/linkpred.pdf)
4. [Project report](https://www.overleaf.com/project/63c6ce9ebfd91ea9e32541d1)

### Contributors:
1. William Howard-Snyder
2. Pallavi Banerjee
3. Therese Pena Pacio
4. Hans Easton

### Setup Instructions:
1. Clone the repo onto your machine with `git clone https://github.com/hanseaston/LinkPredictionOGB.git`
2. Navigate to the directory with `cd LinkPredictionOGB`
3. Create a new conda environment `conda create --name link-pred --file conda_requirements.txt`
4. Some OGB packages are only available on `pip` so also run `pip install -r pip_requirements.txt` 
5. Explore with `python main.py`


### Directory Structure:
1. dataset/
    - module for data pre-processing and analysis
    - direcotry for storing other data related info on disk (e.g., original graph)
2. experiments/
    - module for experiment files (training, testing)
3. models/
    - module for model definition files
4. results/
    - directory for storing link prediction results for each model/trial