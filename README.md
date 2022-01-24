# KerGNNs: Interpretable Graph Neural Networks with Graph Kernels
This repository is the official PyTorch implementation of "KerGNNs: Interpretable Graph Neural Networks with Graph Kernels", Aosong Feng, Chenyu You, Shiqiang Wang, Leandros Tassiulas, AAAI 2022 [link] (https://arxiv.org/abs/2201.00491).


## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 1.6.0 version.

Then install the other dependencies.
```
pip install -r requirements.txt
```
## Datasets
All the dataset are downloaded from
```bash
https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
```
The data spilts are the same with [here](https://github.com/diningphil/gnn-comparison).
## Test Run
- 1-layer KerGNN with random walk graph kernel, IMDB-BINARY dataset
```bash
python main.py --iter 0
```
- 2-layer KerGNN with deep random walk kernel, PROTEINS-full dataset
```bash
python main.py --iter 0 --dataset PROTEINS_full --kernel drw --hidden_dims 0 16 32 --size_graph_filter 4 4 --use_node_labels --no_norm
```
