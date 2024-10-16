# # import networkx as nx
# # from pynauty import Graph as NautyGraph, canon_label

# # def networkx_to_nauty(graph):
# #     adjacency_dict = {node: set(neighbors) for node, neighbors in graph.adjacency()}
# #     return NautyGraph(len(graph.nodes()), adjacency_dict)

# # def canonical_form(graph):
# #     nauty_graph = networkx_to_nauty(graph)
# #     canonical_label = canon_label(nauty_graph)
# #     return canonical_label


# # n = 100
# # p = 0.2
# # G = nx.erdos_renyi_graph(n, p)

# # canonical_label = canonical_form(G)
# # print("Canonical Label:", canonical_label)


# import networkx as nx
# from pynauty import Graph as NautyGraph, canon_label

# def networkx_to_nauty(graph):
#     adjacency_dict = {node: set(neighbors) for node, neighbors in graph.adjacency()}
#     return NautyGraph(number_of_vertices=len(graph.nodes()), adjacency_dict=adjacency_dict)

# def canonical_form(graph):
#     nauty_graph = networkx_to_nauty(graph)
#     canonical_label = canon_label(nauty_graph)
#     print('before:', canonical_label)
#     # Create a mapping from original node to canonical node
#     node_mapping = {original: canonical for canonical, original in enumerate(canonical_label)}
    
#     # Create a new graph with the canonical node ordering
#     canonical_graph = nx.Graph()
#     canonical_graph.add_nodes_from(range(len(canonical_label)))
    
#     for u, v in graph.edges():
#         canonical_graph.add_edge(node_mapping[u], node_mapping[v])
    
#     nauty_graph = networkx_to_nauty(canonical_graph)
#     canonical_label = canon_label(nauty_graph)
#     print('after:', canonical_label)

#     return canonical_graph

# def are_isomorphic(graph1, graph2):
#     return nx.is_isomorphic(graph1, graph2)

# # Example usage:
# def main():
#     # Define two different graphs
#     G1 = nx.Graph()
#     G1.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3)])



#     G2 = nx.Graph()
#     G2.add_edges_from([(2, 1), (1, 3), (3, 2), (2, 0)])
#     # G2.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3)])

#     # Compute canonical forms
#     canonical_graph_1 = canonical_form(G1)
#     print(canonical_graph_1.edges())
#     canonical_graph_2 = canonical_form(G2)
#     print(canonical_graph_2.edges())

#     # Compare canonical forms
#     if are_isomorphic(canonical_graph_1, canonical_graph_2):
#         print("The graphs are isomorphic.")
#     else:
#         print("The graphs are not isomorphic.")

# if __name__ == "__main__":
#     main()


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_2d_mesh_graph(size):
    """Generates a 2D mesh graph of given size."""
    G = nx.grid_2d_graph(size, size)
    return nx.convert_node_labels_to_integers(G)

def adjacency_matrix_to_npy(G, filename):
    """Saves the adjacency matrix of graph G to a .npy file."""
    adj_matrix = nx.to_numpy_array(G)
    np.save(filename, adj_matrix)

def permute_graph(G):
    """Generates a random permutation of the graph G."""
    nodes = list(G.nodes())
    permutation = np.random.permutation(nodes)
    mapping = {old: new for old, new in zip(nodes, permutation)}
    return nx.relabel_nodes(G, mapping)

# Generate a 2D mesh graph
size = 4  # You can change this to any size
G1 = generate_2d_mesh_graph(size)

# Generate a permuted (isomorphic) graph
G2 = permute_graph(G1)

# Save adjacency matrices to .npy files
adjacency_matrix_to_npy(G1, 'graph1.npy')
adjacency_matrix_to_npy(G2, 'graph2.npy')

# Optionally, visualize the graphs
plt.figure(figsize=(10, 5))

plt.subplot(121)
nx.draw(G1, with_labels=True, node_color='lightblue')
plt.title("Graph 1")

plt.subplot(122)
nx.draw(G2, with_labels=True, node_color='lightgreen')
plt.title("Graph 2")

plt.show()
