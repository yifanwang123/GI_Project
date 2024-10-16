from utils2 import Candidate, Partition, TreeNode
from SS_al2 import initialize_schreier_sims_with_partitions
from sympy.combinatorics import Permutation
import sys
from collections import defaultdict
import copy
import itertools
import networkx as nx
import matplotlib.pyplot as plt


def individualize(partition, vertex):
    n = len(partition.cls)

    new_cls = copy.deepcopy(partition.cls)
    new_inv = copy.deepcopy(partition.inv)
    vertex_class_current = partition.cls[vertex]
    # print(f'individulized node cell id: {vertex_class_current}')

    
    for node in range(n):
        if partition.cls[node] < vertex_class_current or node == vertex:
            new_cls[node] = partition.cls[node]
        else:
            new_cls[node] = partition.cls[node] + 1
        if node in partition.cell_dict[vertex_class_current] and node != vertex:
            new_inv[node] += 1

    cells = partition.cells + 1
    new_dict = {}
    
    # for i in range(n):
    #     if new_cls[i] not in new_dict:
    #         new_dict[new_cls[i]] = [i]
    #     else:
    #         new_dict[new_cls[i]].append(i)

    # new_dict = {key: new_dict[key] for key in sorted(new_dict.keys())}
    curr_dict = copy.deepcopy(partition.cell_dict)
    curr_lst = [[]] * partition.cells
    for idx, value in curr_dict.items():
        curr_lst[idx] = value

    indi_label = find_key_for_element(curr_dict, vertex)
    a = curr_dict[indi_label]
    # print(a)
    a.remove(vertex)
    # print(a)
    curr_lst = split_and_update_list(curr_lst, indi_label, [[vertex], a])
            
    for i, lst in enumerate(curr_lst):
        new_dict[i] = lst


    new_partition = Partition(cls=new_cls, inv=new_inv, cell_dict=new_dict, active=0, cells=cells, code=partition.code)
    # print(new_partition.cell_dict)
    # print(new_partition.cls)
    return new_partition


# def one_dimensional_refinement(graph, partition):
#     n = graph.num_nodes
#     cls_ = partition.cls
#     inv = partition.inv

#     refined = True
#     while refined:
#         refined = False
#         new_cls = [0] * n
#         new_inv = [0] * n
#         neighbor_classes = {}

#         for i in range(n):
#             neighbors = tuple(sorted(cls_[neighbor] for neighbor in graph.neighbor_dict[i]))
#             if neighbors not in neighbor_classes:
#                 neighbor_classes[neighbors] = len(neighbor_classes)
#             new_cls[i] = neighbor_classes[neighbors]
#             new_inv[new_cls[i]] = i

#         if new_cls != cls_:
#             refined = True
#             cls_ = new_cls
#             inv = new_inv

#     partition.cls = cls_
#     partition.inv = inv
#     return partition

# def two_dimensional_refinement(graph, partition):
#     n = graph.num_nodes
#     cls = partition.cls
#     inv = partition.inv

#     refined = True
#     while refined:
#         refined = False
#         new_cls = [0] * n
#         new_inv = [0] * n
#         neighbor_pairs = {}

#         for i in range(n):
#             for j in graph.neighbor_dict[i]:
#                 pair = (cls[i], cls[j])
#                 if pair not in neighbor_pairs:
#                     neighbor_pairs[pair] = len(neighbor_pairs)
#                 new_cls[i] = neighbor_pairs[pair]

#         unique_classes = len(neighbor_pairs)
#         if unique_classes > len(new_inv):
#             new_inv = [0] * unique_classes

#         for i in range(n):
#             new_inv[new_cls[i]] = i

#         if new_cls != cls:
#             refined = True
#             cls = new_cls
#             inv = new_inv

#     partition.cls = cls
#     partition.inv = inv
#     return partition



def node_neighbors_color(node_neighbors, cls):
    color_assign = tuple(sorted(cls[neighbor] for neighbor in node_neighbors))
    # print('here!!:', color_assign)
    return color_assign



def find_outer_index(list_of_lists, element):
    for outer_index, inner_list in enumerate(list_of_lists):
        if element in inner_list:
            return outer_index
    return -1



def reorder_graph_based_on_neighbor_degrees(G):
    neighbor_degrees = {node: sum(G.degree(neighbor) for neighbor in G.neighbors(node)) for node in G.nodes()}
    
    sorted_nodes = sorted(G.nodes(), key=lambda x: neighbor_degrees[x], reverse=True)
    
    H = nx.Graph()
    mapping = {node: idx for idx, node in enumerate(sorted_nodes)}
    H.add_nodes_from(range(len(sorted_nodes)))
    H.add_edges_from((mapping[u], mapping[v]) for u, v in G.edges())
    
    return H


# def one_dimensional_refinement(G, partition, node_invariant):
    
#     n = G.num_nodes
#     # print(n)
#     curr_cls = copy.deepcopy(partition.cls)


#     # a = {}
#     # for i in range(n):
#     #     a[i] = curr_cls[i]
#     # draw_graph(G.graph, a)
    

#     sorted_dict = {k: v for k, v in sorted(partition.cell_dict.items(), key=lambda item: len(item[1]))}
#     print(f'sorted_dict: {sorted_dict}')
#     for cell_id, cell in sorted_dict.items():
        
#         if len(cell) == 1:
#             continue
        
#         cell = node_invariant.node_each_cell_invariant(cell, curr_cls)
        
#         # print('here!!:', cell)

#         neighbors_colors = {node: node_neighbors_color(G.neighbor_dict[node], curr_cls) for node in cell}
#         # print(f'neighbors_colors of cell {cell_id}: {neighbors_colors}')

#         new_color_dict = defaultdict(list)
#         # print(new_color_dict)
#         for node, color in neighbors_colors.items():
#             new_color_dict[color].append(node)
        
#         new_color_pattern = list(new_color_dict.values())
#         # print(new_color_pattern)

#         # print('end')
#         if len(new_color_pattern) > 1:
#             new_cls = copy.deepcopy(curr_cls)

#             # print(f'here 1: {curr_cls}', new_color_pattern)
#             # for node in range(n):
#             #     node_curr_cell_id = curr_cls[node]
#             #     if node in cell:
#             #         # print(f'here2: node {node}, cell {cell}')
#             #         idx = find_outer_index(new_color_pattern, node)
#             #         new_cls[node] = node_curr_cell_id + idx
#             #     else:
#             #         # print('here3:', node, node_curr_cell_id, curr_cls[new_color_pattern[0][0]])
#             #         if node_curr_cell_id > curr_cls[new_color_pattern[0][0]]:
#             #             new_cls[node] = node_curr_cell_id + len(new_color_pattern) - 1
#             # curr_cls = new_cls
#         # print(f'here4: {curr_cls}')
#             # cell = node_invariant.node_each_cell_invariant(cell)
            
#             for node in cell:
#                 node_curr_cell_id = curr_cls[node]
#                 idx = find_outer_index(new_color_pattern, node)
#                 new_cls[node] = node_curr_cell_id + idx
#             for node in range(n):
#                 if node not in cell:
#                     node_curr_cell_id = curr_cls[node]
#                         # print('here3:', node, node_curr_cell_id, curr_cls[new_color_pattern[0][0]])
#                     if node_curr_cell_id > curr_cls[new_color_pattern[0][0]]:
#                         new_cls[node] = node_curr_cell_id + len(new_color_pattern) - 1
#             curr_cls = new_cls



#     # print('finish for loop')
#     new_cells = len(set(curr_cls))
#     new_inv = [float('-inf')] * n
    
#     new_dict = {}

#     for node in range(n):
#         if curr_cls[node] not in new_dict:
#             new_dict[curr_cls[node]] = [node]
#         else:
#             new_dict[curr_cls[node]].append(node)

#     lst = [0] * new_cells


#     for cell in new_dict.keys():
#         lst[cell] = len(new_dict[cell])

#     for i in range(n):
#         cell_id = curr_cls[i]
#         new_inv[i] = 1 + sum(lst[:cell_id])

#     code = node_invariant.compute_invariant(curr_cls)

#     new_partition = Partition(cls=curr_cls, inv=new_inv, cell_dict=new_dict, 
#                                         active=1, cells=new_cells, code=code)
    
    
#     # print('finish_one_ref:', new_partition.cell_dict)
#     return new_partition




def num_edge(G, node, target_cell):
    num = 0
    
    for target_node in target_cell:
        
        if G.graph.has_edge(node, target_node):
            # print(node, target_node)
            num += 1
    return num



def find_element_indices(element, list_of_lists):
    # indices = []
    for idx, inner_list in enumerate(list_of_lists):
        if element in inner_list:
            return idx
    # return indices

def has_singleton(sequence):
    for sublist in sequence:
        if len(sublist) == 1:
            sequence.remove(sublist)
            return sublist
    return sequence.pop()



def update_dict_with_split_lists(d, key_to_split, split_lists):
    new_dict = {}
    current_index = 0

    for key in sorted(d.keys()):
        if key == key_to_split:
            for split_list in split_lists:
                new_dict[current_index] = split_list
                current_index += 1
        else:
            new_dict[current_index] = d[key]
            current_index += 1

    return new_dict


def split_and_update_list(lst, index_to_split, split_lists):
    
    original_list_to_split = lst.pop(index_to_split)
    
    for i, split_list in enumerate(split_lists):
        lst.insert(index_to_split + i, split_list)
    
    return lst


def find_key_for_element(d, element):
    for key, value_list in d.items():
        if element in value_list:
            return key
    


def add_all_but_one_largest_list_lexicographically(a, b):
    # if not a:
    #     return b
    
    # largest_sublist = max(a)
    
    # count_largest = a.count(largest_sublist)
    
    # for sublist in a:
    #     if sublist == largest_sublist and count_largest > 0:
    #         count_largest -= 1
    #     else:
    #         b.append(sublist)
    
    if not a:
        return b
    
    largest_sublist = max(a, key=len)
    
    count_largest = a.count(largest_sublist)
    
    for sublist in a:
        if sublist == largest_sublist and count_largest > 0:
            count_largest -= 1
        else:
            b.append(sublist)
    
    
    return b

def find_sublist_index(list_of_sublists, element):
    for index, sublist in enumerate(list_of_sublists):
        if element in sublist:
            return index
    return -1 


def one_dimensional_refinement_helper(G, partition, sequence, node_invariant):
    n = G.num_nodes
    
    
    num = 0
    while sequence and not is_discrete(partition):
        # print('===============================================================')
        curr_dict = copy.deepcopy(partition.cell_dict)
        curr_lst = [[]] * partition.cells
        for idx, value in curr_dict.items():
            curr_lst[idx] = value
        num += 1

        curr_cls = [0] * n
        for i in range(n):
            curr_cls[i] = find_sublist_index(curr_lst, i)
        # print(sequence)

        target_cell = has_singleton(sequence)
        # print(curr_lst)
        
        # print(f'target_cell: {target_cell}')
        # print('-------------------------')
        for cell_id, cell in curr_dict.items():
            # print(f'current cell: {cell_id}:{cell}')
            
            cell_new = node_invariant.node_each_cell_invariant(cell, curr_cls)
            dict_ = {}
            for node in cell_new:
                num_edges = num_edge(G, node, target_cell)
                if num_edges in dict_:
                    dict_[num_edges].append(node)
                else:
                    dict_[num_edges]=[node]

            values = list(dict_.values())
            # print(dict_)

            key_sublist_pairs = [(key, sublist) for key, sublist in dict_.items() if sublist in values]

            # Sort the list of tuples first by the length of the sublist, then by the key
            sorted_key_sublist_pairs = sorted(key_sublist_pairs, key=lambda x: (len(x[1]), x[0]))

            # Extract the sorted sublists
            values = [sublist for key, sublist in sorted_key_sublist_pairs]




            # print(f'values: {values}')
            # print(dict_)
            # sys.exit()
            # new_dict = update_dict_with_split_lists(curr_dict, find_key_for_element(curr_dict, cell), values)
            # # print(new_dict)
            # partition.cell_dict = new_dict
            # partition.cells = len(partition.cell_dict)
            # print(partition.cell_dict)
            # print(f'current num cells: {partition.cells}')
            # for node in range(n):
            #     partition.cls[node] = find_key_for_element(partition.cell_dict, node)
            # print(f'current cls: {partition.cls}')


            curr_lst = split_and_update_list(curr_lst, curr_lst.index(cell), values)
            # print(f'current partition: {curr_lst}')
            partition.cells = len(curr_lst)
            # print(f'current num cells: {partition.cells}')
            # sys.exit()

            
            # print(f'before sequence:{sequence} ')

            if cell in sequence:
                # print('here')
                idx = sequence.index(cell)
                sequence = split_and_update_list(sequence, idx, values)
                # print(f'current sequence: {sequence}')

            else:
                # print('here1')
                sequence = add_all_but_one_largest_list_lexicographically(values, sequence)
                # print(f'current sequence: {sequence}')
            # print(dict_)
            # print(values)
            # print(f'current sequence: {sequence}')
        
        
        new_dict = {}
        for idx, lst in enumerate(curr_lst):
            new_dict[idx] = lst
        partition.cell_dict = new_dict
        for node in range(n):
            partition.cls[node] = find_key_for_element(partition.cell_dict, node)

        # print(f'current cls: {partition.cls}')
        # print(f'current dict: {partition.cell_dict}')

        # if num == 3:
        #     sys.exit()


            # for node in range(G.num_nodes):
            #     if node in cell:
            #         curr_cls[node] = cell_id + find_element_indices(node, values)
            #     elif curr_cls[node] > cell_id:
            #         curr_cls[node] = curr_cls[node] + len(dict_)
                

    new_inv = [float('-inf')] * n
    lst = [0] * partition.cells


    for cell in partition.cell_dict.keys():
        lst[cell] = len(partition.cell_dict[cell])

    for i in range(n):
        cell_id = partition.cls[i]
        new_inv[i] = 1 + sum(lst[:cell_id])

    partition.inv = new_inv
    # print(f'current inv: {partition.inv}')

    partition.code = node_invariant.compute_invariant(partition.cls)
    partition.active = 1
    # print('Final ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # print(partition.cell_dict)
    # print(partition.cells)
    # print(partition.cls)
    # print(partition.inv)
    # sys.exit()

    return partition






def one_dimensional_refinement(G, partition, node_invariant, individulized_node=None):
    
    n = G.num_nodes
    
    

    # sorted_dict = {k: v for k, v in sorted(partition.cell_dict.items(), key=lambda item: len(item[1]))}
    if individulized_node == None:
        new_partition = one_dimensional_refinement_helper(G, partition, [partition.cell_dict[0]], node_invariant)
        # new_partition.code = node_invariant.compute_invariant(new_partition.cls)
    else:

        new_partition = one_dimensional_refinement_helper(G, partition, [[individulized_node]], node_invariant)
        # new_partition.code = node_invariant.compute_invariant(new_partition.cls)

    return new_partition













def one_dimensional_refinement_2(G, partition, node_invariant):
    
    
    new_partition = one_dimensional_refinement(G, partition, node_invariant)
    # while not is_equitable(G, new_partition):
    
    #     new_partition = one_dimensional_refinement(G, partition, node_invariant)
        
    return new_partition



def is_equitable(G, partition):
    for cell in range(partition.cells):
        nodes = partition.cell_dict[cell]
        for node_i, node_j in itertools.combinations(nodes, 2):
            neighbor_i = G.neighbor_dict[node_i]
            neighbor_j = G.neighbor_dict[node_j]
            node_i_color = node_neighbors_color(neighbor_i, partition.cls)
            # print(f'here6: {node_i_color}')
            node_j_color = node_neighbors_color(neighbor_j, partition.cls)
            # print(f'here7: {node_j_color}')

            if node_i_color != node_j_color:
                return False
    return True



def k_dimensional_refinement(G, partition, generators, node_invariant):

    partition = one_dimensional_refinement(G, partition, node_invariant)
    queue = [partition]
    
    while queue:
        current_partition = queue.pop(0)
        # print(f'what! {current_partition.cls}')

        
        max_count = -1
        target_cell = -1
        current_partition = copy.deepcopy(current_partition)

        for idx, cell in current_partition.cell_dict.items():
            count = 0
            if len(cell) == 1:
                continue
            for node in cell:
                # print(f'current: ', current_partition.cls)
                # print(f'individulized node', node, cell)
                
                new_partition = individualize(current_partition, node)
                # print(current_partition.cell_dict)
                # print(new_partition.cell_dict)
                # sys.exit()
                new_partition = one_dimensional_refinement(G, new_partition, node_invariant)
                # print(new_partition.cls)
                # print(new_partition.cell_dict)
                if is_discrete(new_partition):
                    count += 1
                    update_generators(generators, G)
                else:
                    queue.append(new_partition)
            # print('shenme!', len(queue))
            if count > max_count:
                max_count = count
                target_cell = idx
        
    return new_partition

def is_discrete(partition):
    # print('haaaaaaaaaa', partition.cells, len(partition.cls))
    return partition.cells == len(partition.cls)


# def update_generators(generators, graph):
#     current_generators = [Permutation(p) for p in generators]
#     new_generators = random_schreier_sims(generators, graph.num_nodes)
#     generators.extend(new_generators)


def symmetry_pruning(traces, current_trace, generators):

    # for generator in generators:
    #     perm = Permutation(generator)
    #     if apply_automorphism(trace, current_trace, perm):
    #         return True
    
    for g in generators:
        for trace in traces:
            if are_sequences_equinvariant(trace, current_trace, g):
                return True
    return False

def apply_automorphism(trace, current_trace, generator):
    index_map = list(generator)
    permuted_sequence = [trace[index_map[i]] for i in range(len(trace))]
    return permuted_sequence == current_trace


def are_sequences_equinvariant(seq1, seq2, g):
    if len(seq1) != len(seq2):
        return False
    transformed_seq1 = apply_automorphism(seq1, g)
    return transformed_seq1 == seq2


def apply_automorphism(sequence, g):
    return tuple(g[v] for v in sequence)


    # new_cls = partition.cls[:]

    # for i in range(len(generator)):
    #     new_cls[i] = partition.cls[generator[i]]

    # new_partition = Partition(cls=new_cls, inv=partition.inv, cell_dict=partition.cell_dict, active=partition.active, cells=partition.cells, code=partition.code)
    # return new_partition



def invariant_pruning(partition, invariants, node_invariant):
    invariant_code = compute_invariant(partition, node_invariant)
    if invariant_code in invariants:
        return True
    invariants.add(invariant_code)
    return False


def compute_invariant(partition, node_invariant):
    return node_invariant.compute_invariant(partition)


def select_target_cell(partition, node_invariant):
    # target_cell = None
    # max_size = -1
    # # print(partition.cell_dict)
    # # sys.exit()
    # for key, value in partition.cell_dict.items():
    #     curr_size = len(value)
    #     if curr_size > max_size:
    #         target_cell = key
    #         max_size = curr_size
    #     elif curr_size == max_size:
    #         # print('here')
    #         curr_cell = node_invariant.node_each_cell_invariant(value, partition.cls)
    #         target_cell_items = node_invariant.node_each_cell_invariant(partition.cell_dict[target_cell], partition.cls)

    #         if curr_cell > target_cell_items:
    #             # print('here2')
    #             target_cell = key
    #             max_size = curr_size

    if not partition.cell_dict:
        return None
    invariant = []
    for key, value in partition.cell_dict.items():
        cell_inv = node_invariant.cell_invariant(value, partition.cls)
        invariant.append(cell_inv)

    max_length = max(len(v) for v in partition.cell_dict.values())
    keys_with_max_length = [k for k, v in partition.cell_dict.items() if len(v) == max_length]
    if len(keys_with_max_length) == 1:
        return keys_with_max_length[0] 
    max_property = tuple()
    selected_key = None
    for key in keys_with_max_length:
        property_value = invariant[key]
        if property_value > max_property:
            max_property = property_value
            selected_key = key

    return selected_key
    


def prune_pa(nodes_to_refine, current_invariant):
    # print('here a')
    for node in nodes_to_refine:
        if node.invariant < current_invariant:
            nodes_to_refine.remove(node)

def prune_pb(nodes_to_refine, current_invariant):
    # print('here b')

    for node in nodes_to_refine:
        if node.invariant != current_invariant:
            nodes_to_refine.remove(node)


def find_node_largest_invariant(partition, target_cell, node_invariant):
    best_node = None
    largest = None
    best_partition = None
    for node in partition.cell_dict[target_cell]:
        current_partition = individualize(partition, node)
        curr_node_invariant = node_invariant.compute_invariant(current_partition)
        if largest is None or curr_node_invariant > largest:
            largest = curr_node_invariant
            best_node = node  
            best_partition = current_partition

    return best_node, best_partition




def draw_graph(graph, labels=None, title="Graph"):
    pos = nx.spring_layout(graph)  # positions for all nodes
    plt.figure() 

    nx.draw(graph, pos, with_labels=True, labels={node: node for node in graph.nodes()}, node_color='lightblue', node_size=500, font_size=10, font_color='black', edge_color='gray')
    if labels:
        labels_pos1 = {node: (pos[node][0], pos[node][1] + 0.05) for node in pos}
        nx.draw_networkx_labels(graph, labels_pos1, labels=labels, font_size=12, font_color='red')
    # ax.set_title(title1)

    plt.title(title)
    plt.show()



def refine_partition(graph, partition, node_invariant, seen_codes, record_file):
    
    
    partition = one_dimensional_refinement(graph, partition, node_invariant)
    # print('At beginning')
    # # print(partition.cls)
    # print(partition.cell_dict)
    # partition.code = 
    a = {}
    for i in range(graph.num_nodes):
        a[i] = partition.cls[i]
    # draw_graph(graph.graph, a)
    # sys.exit()
    
    traces = []
    trace = []
    root = TreeNode(partition, trace)
    root.invariant = partition.code
    # print('here')
    # print(root.partition.cls)
    current_level_nodes = [root]
    next_level_nodes = []
    leaves = []
    level = 0
    generators = []

    f = open(record_file, "w")
    

    while current_level_nodes:
        print('======================================================')

        # print('New level', level, len(current_level_nodes))
        for current_node in current_level_nodes:
            # print(f'current level {level}, current node {current_level_nodes.index(current_node)}')
            current_partition = current_node.partition
            current_invariant = current_node.invariant
            # print('here!!!!', len(current_node.sequence))
            
            if is_discrete(current_node.partition):
                # print('discrete node idx', current_level_nodes.index(current_node))
                traces.append(current_node.sequence)
                leaves.append(current_node)
                if len(leaves) == 2:
                    a, b = leaves[0], leaves[1]
                    schreier_sims = initialize_schreier_sims_with_partitions(a.partition, b.partition)
                    continue
                elif len(leaves) !=1:
                    schreier_sims.update_with_partition(current_partition)
                    generators = schreier_sims.get_generators()
                    # print('here', generators)
                    continue
                else:
                    continue

            if len(leaves) > 2:
                if symmetry_pruning(traces, current_node.sequence, generators):
                    print('here!!!!')
                    continue
            
            prune_pa(current_level_nodes, current_invariant)
            prune_pb(current_level_nodes, current_invariant)


            # if current_invariant not in seen_codes:
            #     seen_codes.add(current_invariant)
            # print('current partition', current_partition.cell_dict, f'current node: {current_node.sequence}')
            target_cell = select_target_cell(current_partition, node_invariant)
            # print(f'target_cell: {target_cell}', len(current_partition.cell_dict[target_cell]))
            # print(current_partition.cell_dict[target_cell])
            a = node_invariant.node_each_cell_invariant(current_partition.cell_dict[target_cell], current_partition.cls)
            print(f'level {level}: {a[0]}')
            f.write(f'level {level}: {a[0]} \n')
            # sys.exit()
            # print(len(current_node.sequence), f'target cell size {len(a)}', f'current num leaves {len(leaves)}')
            # print()
            
            for node in a:
                temp_invariant = current_node.invariant
                temp_trace = copy.deepcopy(current_node.sequence)
                temp_trace.append(node)
                indi_partition = individualize(current_partition, node)
                # if node == 4 or node ==19:
                
                # print(f'after indi {indi_partition.cell_dict}')
                partition = one_dimensional_refinement(graph, indi_partition, node_invariant, individulized_node=node)
                # if node == 4 or node ==19:
                if not is_discrete(partition):
                    print(f'temp trace {temp_trace}')
                    print(f'after refin {partition.cell_dict}')
                # if node == 0 or node == 3:
                #     sys.exit()
                child_node = TreeNode(partition, parent=current_node, sequence=temp_trace)
                child_node.invariant = temp_invariant + partition.code
                current_node.add_child(child_node)
                next_level_nodes.append(child_node)
        # print('======================================================')
        
        level += 1
        current_level_nodes = next_level_nodes
        next_level_nodes = []
    f.close()
    return root, leaves, generators

    