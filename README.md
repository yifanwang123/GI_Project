# Graph Isomorphism Project
This repository contains research code for graph isomorphism testing. It implements the **Tracy algorithm** for identifying whether two graphs are isomorphic. The core idea is to generate a canonical labelling of each graph via a learned representation. Two graphs are deemed isomorphic if their canonical representations coincide. The implementation draws heavily on PyTorch Geometric, neural networks tailored to graphs (e.g., Graph Isomorphism Networks and Graph Attention Networks), and custom differentiable operators that mimic discrete step functions. A Siamese architecture is trained to produce embeddings whose L2 distance is small for isomorphic pairs and large for non-isomorphic pairs.

## Prerequisites

This project requires **Python ≥ 3.8** and the following Python packages:

* [`torch`](https://pytorch.org/) ≥ 1.10 with CUDA support if training on a GPU.
* [`torch_geometric`](https://pytorch-geometric.readthedocs.io/) and its dependencies (`torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`).  Follow the [official installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) because wheels depend on your CUDA and PyTorch versions.
* [`numpy`](https://numpy.org/) and [`scipy`](https://www.scipy.org/).
* [`networkx`](https://networkx.org/) for graph parsing and manipulation.
* [`tqdm`](https://tqdm.github.io/) for progress bars.

You may also need `matplotlib` if you wish to visualise graphs or intermediate results.

## Datasets
The project supports multiple datasets:

[benchmark1](https://www.lics.rwth-aachen.de/go/id/rtok/) – A set of paired graphs stored in plain text .npz files. The load_dataset_benchmark1.py script reads each graph, generates a corresponding record file, pairs graphs with their isomorphic counterparts and labels them accordingly.

[EXP / CEXP](https://arxiv.org/abs/2010.01179) – Experimental graphs where each graph is stored as a NumPy adjacency matrix. Use load_EXP_CEXP.py to create graph pairs and labels.

self_generated_data – Synthetic graphs created by the author. Use load_dataset_self_generated.py to generate graph pairs and labels. A PDF (self_generated_2_exact.pdf) documents the exact labelling process for a set of synthetic graphs.

Each dataset yields a collection of samples of the form (graph_1_data, graph_2_data, list_of_matrix_1, list_of_matrix_2, idx1, idx2) plus a binary label indicating whether the graphs are isomorphic. These samples are serialised with torch.save and pickle.dump and loaded on the fly by train.py

## Run
```
python train.py \
    --dname benchmark1 \        # dataset name; must match folder under `data/`
    --GIN_num_layers 2 \        # number of GIN layers in the encoder
    --GIN_hidden_dim 8 \        # hidden dimension size
    --epochs 500 \              # number of training epochs
    --runs 5 \                  # number of random restarts
    --lr 0.01 \                 # learning rate
    --weight_decay 0.05 \       # weight decay (L2 regularisation)
    --numL 5 \                  # number of nodes to sample for local information
    --numK 5 \                  # number of partitions in ExactModel
    --reg_lambda 0.01 \         # regularisation term for the ExactModel
    --cuda 0                    # GPU id (set to -1 for CPU)
```
