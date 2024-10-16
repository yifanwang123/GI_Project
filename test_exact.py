from isgnn import ExactModel
import torch
from torch_geometric.data import Data
import networkx as nx
import sys
import numpy as np
import matplotlib.pyplot as plt


def check_iso(form1, form2, graph1, graph2):

    iso = True
    adj_matrix_1 = nx.adjacency_matrix(graph1)
    adj_matrix_dense_1 = adj_matrix_1.todense()
    # print(adj_matrix_dense)
    adj_matrix_2 = nx.adjacency_matrix(graph2)
    adj_matrix_dense_2 = adj_matrix_2.todense()

    dt_1 = {}
    dt_2 = {}
    for i, row in enumerate(adj_matrix_dense_1):
        num_neighbors_1 = np.sum(row)
        num_neighbors_2 = np.sum(adj_matrix_dense_2[i])
        if num_neighbors_1 not in dt_1:
            dt_1[num_neighbors_1] = 1
        else:
            dt_1[num_neighbors_1] += 1
        
        if num_neighbors_2 not in dt_2:
            dt_2[num_neighbors_2] = 1
        else:
            dt_2[num_neighbors_2] += 1
        
    if dt_1 != dt_2:    
        print(f"Not iso")
        iso = False

    if iso:
        num_edge = 0
        for u, v in graph1.edges():
            num_edge += 1
            node1 = form2.index(form1[u])
            node2 = form2.index(form1[v])
            if (node1, node2) in graph2.edges():
                continue
            else:
                print(num_edge)
                print((u, v), (node1, node2))
                return False
    
    else:
        for u, v in graph1.edges():
            node1 = form2.index(form1[u])
            node2 = form2.index(form1[v])
            if (node1, node2) in graph2.edges():
                continue
            else:
                return True

    return True




def draw_two_graphs(graph1, graph2, labels1=None, labels2=None, title1="Graph 1", title2="Graph 2"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    pos1 = nx.spring_layout(graph1)
    pos2 = nx.spring_layout(graph2)

    nx.draw(graph1, pos1, with_labels=True, labels={node: node for node in graph1.nodes()}, node_color='lightblue', node_size=500, font_size=10, font_color='black', edge_color='gray', ax=ax1)
    if labels1:
        labels_pos1 = {node: (pos1[node][0], pos1[node][1] + 0.1) for node in pos1}
        nx.draw_networkx_labels(graph1, labels_pos1, labels=labels1, font_size=12, font_color='red', ax=ax1)
    ax1.set_title(title1)

    nx.draw(graph2, pos2, with_labels=True, labels={node: node for node in graph2.nodes()}, node_color='lightgreen', node_size=500, font_size=10, font_color='black', edge_color='gray', ax=ax2)
    if labels2:
        labels_pos2 = {node: (pos2[node][0], pos2[node][1] + 0.1) for node in pos2}
        nx.draw_networkx_labels(graph2, labels_pos2, labels=labels2, font_size=12, font_color='red', ax=ax2)
    ax2.set_title(title2)
    plt.savefig('/home/cds/Documents/Yifan/ICLR25/test_exact.pdf', format='pdf')
    # plt.show()





model = ExactModel()
print("=== Directed Graph Example ===")
edge_index_graph = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7],  # Source nodes
    [6, 6, 6, 7, 7, 7, 7, 6]   # Target nodes
], dtype=torch.long)

# edge_index_graph = torch.tensor([
#     [0, 1, 2],  # Source nodes
#     [1, 2, 0]   # Target nodes
# ], dtype=torch.long)



G = nx.DiGraph()
nodes = [i for i in range(8)]
G.add_nodes_from(nodes)
edges = edge_index_graph.t().tolist()  # Transpose and convert to list of edges
G.add_edges_from(edges)

node_features = []
for node in G.nodes:
    in_neighbors = G.in_degree(node)
    node_features.append(in_neighbors)

node_features = torch.tensor(node_features, dtype=torch.float).view(-1, 1)


data_graph = Data(edge_index=edge_index_graph, num_nodes=8)
data_graph.x = node_features

cl, leaves1 = model.forward(data_graph, device='cpu')
print(cl)



G_2 = nx.DiGraph()

edge_index_graph = torch.tensor([
    [2, 3, 4, 5, 6, 7, 1, 0],  # Source nodes
    [0, 0, 0, 1, 1, 1, 0, 1]   # Target nodes
], dtype=torch.long)


# edge_index_graph = torch.tensor([
#     [1, 2, 0],  # Source nodes
#     [2, 0, 1]   # Target nodes
# ], dtype=torch.long)


nodes = [i for i in range(8)]
G_2.add_nodes_from(nodes)
edges = edge_index_graph.t().tolist()  # Transpose and convert to list of edges
G_2.add_edges_from(edges)

node_features = []
for node in G_2.nodes:
    in_neighbors = G_2.in_degree(node)
    node_features.append(in_neighbors)
print(node_features)

node_features = torch.tensor(node_features, dtype=torch.float).view(-1, 1)


data_graph = Data(edge_index=edge_index_graph, num_nodes=8)
data_graph.x = node_features

cl_2, leaves2 = model.forward(data_graph, device='cpu')
print(cl_2)

# sys.exit()


cl_lst = cl.tolist()
cl_2_lst = cl_2.tolist()


label_1 = {}
label_2 = {}
for i in range(len(cl)):
    label_1[i] = cl_lst[i]
    label_2[i] = cl_2_lst[i]



draw_two_graphs(G, G_2, label_1, label_2)

a = check_iso(cl.tolist(), cl_2.tolist(), G, G_2)
print(a)

# for (partition_1, score_1), (partition_2, score_2) in zip(leaves1.items(), leaves2.items()):
#     print(partition_1.tolist())
#     print(score_1)
#     print(partition_2.tolist())
#     print(score_2)


