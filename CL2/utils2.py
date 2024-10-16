import networkx as nx
import numpy as np
import random
import sys

class Candidate:
    def __init__(self, lab, invlab, singcode, code, sortedlab=False, next=None):
        self.lab = lab
        self.invlab = invlab
        self.singcode = singcode
        self.code = code
        self.sortedlab = sortedlab
        self.next = next


class TracesVars:
    def __init__(self, graph, options=None, stats=None):
        self.graph = graph
        self.options = options
        self.stats = stats
        self.partition = None
        self.node_invariant = None
        self.candidates = None
        self.seen_codes = set()
        self.generators = []
        self.n = graph.num_nodes
        self.mark = 0
        self.stackmark = 0
        self.maxdeg = 0
        self.mindeg = self.n



class NodeInvariant:
    def __init__(self, graph):
        self.graph = graph

    def compute_invariant(self, cls):
        invariant = []
        cell_structure = {}
        for idx, cell_id in enumerate(cls):
            if cell_id not in cell_structure:
                cell_structure[cell_id] = []
            cell_structure[cell_id].append(idx)
        for cell_id in sorted(cell_structure):
            cell = cell_structure[cell_id]
            # sys.exit()
            cell_invariant = self.cell_invariant(cell, cls)
            invariant.append(cell_invariant)
        # return tuple(sorted(invariant))
        return tuple(sorted(invariant))


    def cell_invariant(self, cell, cls):
        # cell = 
        cell_neighbors = []
        for vertex in cell:
            neighbors = self.graph.neighbor_dict[vertex]
            neighbor_degrees = sorted([[len(self.graph.neighbor_dict[neighbor]), cls[neighbor]] for neighbor in neighbors])
            # print(vertex, neighbor_degrees)
            cell_neighbors.append(tuple(neighbor_degrees))
        return tuple(sorted(cell_neighbors))
    
    def helper(self, cell, cls):
        cell_neighbors = {}
        for vertex in cell:
            neighbors = self.graph.neighbor_dict[vertex]
            # lst = [[len(self.graph.neighbor_dict[neighbor], ] for neighbor in neighbors]
            # cls_dict = {}
            # for neighbor in neighbors:
            #     if cls[neighbor] not in cls_dict:
            #         cls_dict[cls[neighbor]] = 1
            #     else:
            #         cls_dict[cls[neighbor]] += 1
            # cls_dict = dict(sorted(cls_dict.items(), key=lambda x: x[1]))
            # # values = list(cls_dict.values())

            neighbor_degrees = sorted([[len(self.graph.neighbor_dict[neighbor]), cls[neighbor]] for neighbor in neighbors])
            cell_neighbors[vertex] = neighbor_degrees
        cell = sorted(cell, key=lambda x: cell_neighbors[x])
        return cell
    
    def node_each_cell_invariant(self, cell, cls):

        new_neighbor_rank = {}

        for vertex in cell:
            # new_neighbor_rank[vertex] = self.helper1(self.graph.neighbor_dict[vertex], cls)
            new_neighbor_rank[vertex] = self.helper(self.graph.neighbor_dict[vertex], cls)

        # print(new_neighbor_rank)
        cell_neighbors = {}
        another = {}
        for vertex in cell:
            neighbor_num_neighbor = [self.helper3(self.graph.neighbor_dict[neighbor]) for neighbor in new_neighbor_rank[vertex]]
            another[vertex] = neighbor_num_neighbor
            neighbor_degrees = sorted([[len(self.graph.neighbor_dict[neighbor]), cls[neighbor]] for neighbor in new_neighbor_rank[vertex]])
            cell_neighbors[vertex] = neighbor_degrees
        cell = sorted(cell, key=lambda x: (cell_neighbors[x], another[x]), reverse=True)
        return cell

    def helper3(self, neighbors):
        lst = []
        for node in neighbors:
            lst.append(len(self.graph.neighbor_dict[node]))

        return sorted(lst)


    # def helper1(self, cell, cls):
    #     nodes = [{}] * len(cell) 
    #     for idx, node in enumerate(cell):

    #         a = [len(self.graph.neighbor_dict[neighbor]) for neighbor in self.graph.neighbor_dict[node]]
    #         b = [cls[neighbor] for neighbor in self.graph.neighbor_dict[node]]
    #         paired_list = list(zip(a, b))
    #         sorted_paired_list = sorted(paired_list, key=lambda x: x[0])
    #         a_, b_ = zip(*sorted_paired_list)

    #         nodes[idx]['node'] = node
    #         nodes[idx]['prop1'] = a_
    #         nodes[idx]['prop2'] = b_


    #     sorted_nodes = sorted(nodes, key=lambda x: (x['prop1'], x['prop2']))
    #     sorted_nodes = [i['node'] for i in nodes]
        
    #     return sorted_nodes



class Partition:
    def __init__(self, cls, inv, cell_dict, active, cells, code):
        '''
        cls: List of class sizes. cls[i] indicates the node i's class id. 
        ind: List of position and coloring mapping.
        cell_dict: cell_dict[i] = [node1, ..., node_k]
        active: Indicates if the partition is active.
        cells: Number of cells in the partition.
        code: A code representing the partition state (node invariant function).
        '''
        self.cls = cls  
        self.inv = inv
        self.cell_dict = cell_dict
        self.active = active
        self.cells = cells
        self.code = code

def initialize_part(graph):
    '''
    Random generated
    '''
    n = graph.num_nodes
    nodes = [i for i in range(n)]
    num_cells = 1
    cell_dict = {}
    # random.shuffle(nodes)
    
    
    # for i in range(num_cells):
    #     cell_dict[i] = [nodes.pop()]

    # for element in nodes:
    #     random.choice(list(cell_dict.values())).append(element)    
    
    cell_dict[0] = nodes
    inv = [float('-inf')] * n
    cls_ = [0] * n

    for cell_id in cell_dict.keys():
        for node in cell_dict[cell_id]:
            cls_[node] = cell_id
    

    lst = [0] * num_cells
    for cell in cell_dict.keys():
        lst[cell] = len(cell_dict[cell])

    for i in range(n):
        cell_id = cls_[i]
        inv[i] = 1 + sum(lst[:cell_id])

    partition = Partition(cls=cls_, inv=inv, cell_dict=cell_dict, active=1, cells=num_cells, code=None)
    print(cell_dict)
    # sys.exit()
    return partition


class LabeledGraph:
    def __init__(self, matrix_file, dataset=None):
        '''
        node index start from 0
        '''
        if dataset == 'self_generated_data':
            self.graph, self.num_nodes = self.read_from_file(matrix_file)
        else:
            if dataset == 'benchmark1':
                self.graph, self.num_nodes = self.read_from_file1(matrix_file)
            else:
                self.graph, self.num_nodes = self.read_from_file2(matrix_file)

        self.neighbor_dict = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}
        # adj_matrix = nx.adjacency_matrix(self.graph)

        # # Convert to a dense format for printing
        # adj_matrix_dense = adj_matrix.todense()
        # print(adj_matrix_dense)
        # for i, row in enumerate(adj_matrix_dense):
        #     if np.sum(row) == 4:
        #         print(f"Node {i}: {row}")
        # print(self.num_nodes)
        # print(self.neighbor_dict)
        # sys.exit()
        # self.colors = range(self.num_nodes)
        
    def read_from_file(self, file_path):
        matrix = np.load(file_path)
        nodes = range(len(matrix))
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val != 0:
                    graph.add_edge(i, j)
        return graph, len(nodes)

    def read_from_file1(self, file_path):
        G = nx.Graph()

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if parts[0] == 'p':
                    continue
                elif parts[0] == 'e':
                    nodev = int(parts[1])-1
                    nodeu = int(parts[2]) -1
                    G.add_edge(nodev, nodeu)
        return G, G.number_of_nodes()
    
    def read_from_file2(self, file_path):
        # G = nx.DiGraph()

        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        first_line = lines[0]
        n = int(first_line.split()[1].split('=')[1])
        
        G = nx.DiGraph()
        
        # Process each line to add edges
        for line in lines[1:]:
            if ':' in line:
                parts = line.split(':')
                node = int(parts[0]) - 1
                neighbors = parts[1].strip().replace('.', '').split()
                neighbors = list(map(int, neighbors))
                for neighbor in neighbors:
                    G.add_edge(node, neighbor-1)
        
        return G, n



    def get_node_labels(self):
        return nx.get_node_attributes(self.graph, 'label')


    
    def can_graph(self, graph):
        self.graph = graph
        self.neighbor_dict = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}
        # print(self.neighbor_dict)



class TreeNode:
    def __init__(self, partition, sequence, parent=None, invariant=None):
        self.partition = partition 
        self.parent = parent  
        self.children = []
        self.sequence = sequence
        self.invariant = invariant

    def add_child(self, child_node):
        self.children.append(child_node)


