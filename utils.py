import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import torch
from torch_geometric.utils import from_networkx
from torch.utils.data import Dataset
# import torch
from scipy.sparse import lil_matrix
# from torch_geometric.data import Data, Dataset


class graph:
    def __init__(self, graph_file, record_file, dataset='benchmark1'):
        prepare_tool = prepare(graph_file, record_file)
        if dataset == 'benchmark1':
            self.nx_graph = prepare_tool.read_graph_file_benchmark1()
            self.max_nodes_idx, self.c_form = prepare_tool.read_record_file()
            self.data = prepare_tool.create_data(self.nx_graph)
            self.num_nodes = self.nx_graph.number_of_nodes()
            if "z" in graph_file:
                self.label = 0
            else:
                self.label = 1
            martix_tool = customized_matrix(self.nx_graph)
            self.list_of_matrix = []
        self_generated_format_data = ['self_generated_data', 'self_generated_data_2', 'EXP', 'CEXP']
            
        if dataset in self_generated_format_data:
            self.nx_graph = prepare_tool.read_graph_file_self_generated_data()
            self.max_nodes_idx, self.c_form = prepare_tool.read_record_file()
            self.data = prepare_tool.create_data(self.nx_graph)
            self.num_nodes = self.nx_graph.number_of_nodes()
            if "non" in graph_file:
                self.label = 0
            else:
                self.label = 1
            martix_tool = customized_matrix(self.nx_graph)
            self.list_of_matrix = []

        
class prepare:
    def __init__(self, graph_file, record):
        self.graph_file = graph_file
        self.record_file = record
        # self.graph = self.read_graph_file_benchmark1()
        # self.max_nodes_idx, self.c_form = self.read_record_file()

    def read_graph_file_benchmark1(self):
        G = nx.DiGraph()

        with open(self.graph_file, 'r') as file:
            for line in file:
                parts = line.split()
                if parts[0] == 'p':
                    continue
                elif parts[0] == 'e':
                    nodev = int(parts[1])-1
                    nodeu = int(parts[2]) -1
                    G.add_edge(nodev, nodeu)

        # num_nodes = G.number_of_nodes()
        # final_graph = graph(G, num_nodes, )

        return G
    
    def read_graph_file_self_generated_data(self):
        matrix = np.load(self.graph_file)
        nodes = range(len(matrix))
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val != 0:
                    graph.add_edge(i, j)

        return graph

    def read_record_file(self):
        max_nodes_idx = []
        c_form = []
        last_level_found = False
        with open(self.record_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "level" in line:
                    last_level_found = True
                    number = int(line.split(':')[-1].strip())
                    max_nodes_idx.append(number)
                elif last_level_found and not line.startswith("level"):
                    # print(line)
                    try:
                        c_form = eval(line.strip()) 
                        break
                    except ValueError:
                        continue
        def remove_duplicates(lst):
            seen = set()
            result = []
            for item in lst:
                if item not in seen:
                    result.append(item)
                    seen.add(item)
            return result
        
        max_nodes_idx = remove_duplicates(max_nodes_idx)
        return max_nodes_idx, c_form

    
    def create_data(self, G):
        node_features = []
        for node in G.nodes:
            in_neighbors = G.in_degree(node)
            # out_neighbors = G.out_degree(node)
            # total_neighbors = len(list(G.neighbors(node)))
            # print(G.neighbors(node))
            
            node_features.append(in_neighbors)
        # print(node_features)
        node_features = torch.tensor(node_features, dtype=torch.float).view(-1, 1)
        data = from_networkx(G)
        data.x = node_features
        # data.y = 1
        return data


class customized_matrix:
    def __init__(self, graph):
        self.graph = graph
        # self.edge_index = self.edge_index
        self.num_nodes = self.graph.number_of_nodes()
        # print(self.num_nodes)
        
        a = self.R1_1()
        b = self.R1_2()
        c = self.R1_3()
        d = self.R1_4()
        e = self.R1_5()

        self.list_of_matrix = [a, b, c, d, e]
        


    def R1_1(self):
        input_size = self.num_nodes
        output_size = 2 * self.num_nodes * self.num_nodes
        results = lil_matrix((input_size, output_size))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                for p in range(2*self.num_nodes):
                    if i == j and p%2 ==0:
                        results[i, j*2*self.num_nodes+p] = 1
                    elif p == 2*i +1:
                        results[i, j*2*self.num_nodes+p] = 1
        sparse_matrix = results.tocsr()
        return sparse_matrix


    
    def R1_2(self):
        input_size = 2 * self.num_nodes * self.num_nodes
        output_size = input_size * 2
        # print(self.num_nodes)
        # print(input_size, output_size)
        results = lil_matrix((input_size, output_size))

        for i in range(self.num_nodes*self.num_nodes):
            for p in range(2):
                for q in range(4):
                    if p == 0:
                        if q == 0 or q == 1:
                            results[2*i+p, 4*i+q] = 1
                        if q == 2:
                            results[2*i+p, 4*i+q] = -1
                    if p == 1:
                        if q == 2 or q == 3:
                            results[2*i+p, 4*i+q] = 1
                        if q == 1:
                            results[2*i+p, 4*i+q] = -1
        sparse_matrix = results.tocsr()
        return sparse_matrix

    def R1_3(self):
        input_size = 4 * self.num_nodes * self.num_nodes
        output_size = 3 * self.num_nodes * self.num_nodes
        results = lil_matrix((input_size, output_size))

        for i in range(self.num_nodes * self.num_nodes):
            for p in range(4):
                for q in range(3):
                    if p == q:
                        if p == 0 or p == 1:
                            results[4*i+p, 3*i+q] = 1
                    elif q == p-1:
                        if p == 2 or p == 3:
                            results[4*i+p, 3*i+q] = 1
        sparse_matrix = results.tocsr()
        return sparse_matrix

    def R1_4(self):
        input_size = 3 * self.num_nodes * self.num_nodes
        output_size = 2 * self.num_nodes * self.num_nodes
        results = lil_matrix((input_size, output_size))

        for i in range(self.num_nodes * self.num_nodes):
            for p in range(3):
                for q in range(2):
                    if q == 0:
                        if p == 0 or p == 1:
                            results[3*i+p, 2*i+q] = 1
                        elif p == 2:
                            results[3*i+p, 2*i+q] = -1
                    elif q == 1:
                        if p == 1 or p == 2:
                            results[3*i+p, 2*i+q] = 1
                        elif p == 0:
                            results[3*i+p, 2*i+q] = -1
        sparse_matrix = results.tocsr()
        return sparse_matrix


    def R1_5(self):
        input_size = 2 * self.num_nodes * self.num_nodes
        output_size = self.num_nodes
        results = lil_matrix((input_size, output_size))

        for i in range(self.num_nodes):
            for j in range(2 * self.num_nodes):
                # for p in range(2):
                if j%2 == 0:
                    results[i * 2 * self.num_nodes + j, i] = 1
        sparse_matrix = results.tocsr()
        return sparse_matrix


def draw_connection(connection_matrix):
    layer_1_size = connection_matrix.shape[0]
    layer_2_size = connection_matrix.shape[1]

    # Create a bipartite graph
    B = nx.Graph()

    # Add nodes for the first layer (e.g., layer 1)
    layer_1_nodes = ["L1_" + str(i) for i in range(1, layer_1_size + 1)]
    B.add_nodes_from(layer_1_nodes, bipartite=0)

    # Add nodes for the second layer (e.g., layer 2)
    layer_2_nodes = ["L2_" + str(i) for i in range(1, layer_2_size + 1)]
    B.add_nodes_from(layer_2_nodes, bipartite=1)

    # Add edges based on the connection matrix with weights
    for i in range(layer_1_size):
        for j in range(layer_2_size):
            if connection_matrix[i, j] != 0:
                B.add_edge(layer_1_nodes[i], layer_2_nodes[j], weight=connection_matrix[i, j])

    # Define evenly spaced positions based on the smaller layer
    max_size = max(layer_1_size, layer_2_size)
    min_size = min(layer_1_size, layer_2_size)

    # Position nodes in two distinct layers with even spacing for the smaller layer
    pos = {}
    if layer_1_size < layer_2_size:
        # Layer 1 has fewer nodes
        pos.update((node, (1, i * max_size / min_size)) for i, node in enumerate(layer_1_nodes))  # Layer 1 nodes
        pos.update((node, (2, i)) for i, node in enumerate(layer_2_nodes))  # Layer 2 nodes
    else:
        # Layer 2 has fewer nodes
        pos.update((node, (1, i)) for i, node in enumerate(layer_1_nodes))  # Layer 1 nodes
        pos.update((node, (2, i * max_size / min_size)) for i, node in enumerate(layer_2_nodes))  # Layer 2 nodes

    # Draw the graph
    plt.figure(figsize=(10, 6))
    edges = B.edges(data=True)
    nx.draw(B, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_color='black', edge_color='gray')
    
    # Draw edge labels to show the weight
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(B, pos, edge_labels=edge_labels)
    
    plt.title("Two-Layer Network Representation with Connection Weights")
    plt.show()




def modify_filename(file_path):
    parts = file_path.rsplit('-', 1)  # Split the file path at the last hyphen
    if len(parts) == 2 and parts[1].isdigit():
        new_file_path = f"{parts[0]}-{int(parts[1]) + 1}"
        return new_file_path
    return file_path

def find_graph_pair(file_path):
    path_parts = file_path.rsplit('/', 1)
    if len(path_parts) == 2:
        modified_filename = modify_filename(path_parts[1])
        new_file_path = f"{path_parts[0]}/{modified_filename}"
        return new_file_path
    return file_path


class GraphPairDataset(Dataset):
    def __init__(self, graph_pairs, labels):
        """
        graph_pairs: List of tuples, where each tuple contains
                     (graph_1_data, graph_2_data, graph_1_list_of_matrix, graph_2_list_of_matrix),
                     and all elements are NumPy arrays.
        labels: List of binary labels corresponding to each graph pair.
        """
        self.graph_pairs = graph_pairs
        self.labels = labels

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.graph_pairs)

    def __getitem__(self, idx):
        # Get the data for a single sample based on the index 'idx'
        graph_1_data, graph_2_data, graph_1_matrix, graph_2_matrix, graph_1_node_idx, graph_2_node_idx = self.graph_pairs[idx]
        label = self.labels[idx]

        # graph_1_data_tensor = torch.from_numpy(graph_1_data).float()
        # graph_2_data_tensor = torch.from_numpy(graph_2_data).float()
        graph_1_nodes_tensor = torch.tensor(graph_1_node_idx).float()
        graph_2_nodes_tensor = torch.tensor(graph_2_node_idx).float()
        # graph_1_matrix_tensor = [torch.from_numpy(matrix).float() for matrix in graph_1_matrix]
        # graph_2_matrix_tensor = [torch.from_numpy(matrix).float() for matrix in graph_2_matrix]
        graph_1_matrix_tensor = [sparse_to_torch(matrix) for matrix in graph_1_matrix]
        graph_2_matrix_tensor = [sparse_to_torch(matrix) for matrix in graph_2_matrix]


        return (graph_1_data, graph_2_data, graph_1_matrix_tensor, graph_2_matrix_tensor, graph_1_nodes_tensor, graph_2_nodes_tensor), torch.tensor(label, dtype=torch.float)


def sparse_to_torch(sparse_matrix):
    # Convert scipy sparse matrix to COO format (required by PyTorch)
    sparse_matrix = sparse_matrix.tocoo()
    
    # Create PyTorch sparse tensor
    # indices = torch.LongTensor([sparse_matrix.row, sparse_matrix.col])
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col))

    # Then convert to a LongTensor
    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(sparse_matrix.data)
    shape = sparse_matrix.shape
    
    sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
    return sparse_tensor


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
                print(f'False, iso: {iso}')
                return False
        print(f'True, iso: {iso}')
        return True
    else:
        for u, v in graph1.edges():
            node1 = form2.index(form1[u])
            node2 = form2.index(form1[v])
            if (node1, node2) in graph2.edges():
                continue
            else:
                print(f'True, iso: {iso}')
                return True
        print(f'False, iso: {iso}')
        return False




def draw_two_graphs(graph1, graph2, labels1=None, labels2=None, title1="Graph 1", title2="Graph 2"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    pos1 = nx.spring_layout(graph1)
    pos2 = nx.spring_layout(graph2)

    nx.draw(graph1, pos1, with_labels=True, labels={node: node for node in graph1.nodes()}, node_color='lightblue', node_size=5, font_size=5, font_color='black', edge_color='gray', ax=ax1)
    if labels1:
        labels_pos1 = {node: (pos1[node][0], pos1[node][1] + 0.1) for node in pos1}
        nx.draw_networkx_labels(graph1, labels_pos1, labels=labels1, font_size=5, font_color='red', ax=ax1)
    ax1.set_title(title1)

    nx.draw(graph2, pos2, with_labels=True, labels={node: node for node in graph2.nodes()}, node_color='lightgreen', node_size=5, font_size=5, font_color='black', edge_color='gray', ax=ax2)
    if labels2:
        labels_pos2 = {node: (pos2[node][0], pos2[node][1] + 0.1) for node in pos2}
        nx.draw_networkx_labels(graph2, labels_pos2, labels=labels2, font_size=5, font_color='red', ax=ax2)
    ax2.set_title(title2)
    plt.savefig('/home/cds/Documents/Yifan/GI_Project/self_generated_2_exact.pdf', format='pdf')
    # plt.show()





# def from_raw_to_data(graph_file, record_file):
#     g = graph(graph_file, record_file)
#     data = g.data
#     max_node_idx, c_form = self.max_nodes_idx, self.c_form 
#     list_of_matrix = self.list_of_matrix




# G = nx.Graph()
# G.add_nodes_from([0, 1])
# G.add_edges_from([(0, 1)])
# matrix_tool = customized_matrix(G)
# for matrix in matrix_tool.list_of_matrix:
#     # print(matrix)
#     draw_connection(matrix)

# sys.exit()




# graph_file = "/home/cds/Documents/Yifan/ICLR25/data/benchmark1/samples_raw/cfi-rigid-r2-0072-01-1"
# record_file = "/home/cds/Documents/Yifan/ICLR25/data/benchmark1/samples_raw/cfi-rigid-r2-0072-01-1_record.txt"
# Graph = graph(graph_file, record_file)
# max_nodes_idx, c_form = Graph.max_nodes_idx, Graph.c_form
# num_nodes = Graph.num_nodes
# list_matrix = Graph.list_of_matrix
# data = Graph.data
# print(max_nodes_idx)
# print(c_form)
# print(num_nodes)
# print(data)
# print(list_matrix)