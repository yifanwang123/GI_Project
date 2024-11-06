import torch
import pickle
import os
import time
from utils import graph, find_graph_pair




def save_samples(samples_list, filename):
    # Prepare a list of graph data and a list of other elements
    graph_data_list = []
    other_data_list = []
    
    # Separate the graph data and the other components for each sample
    for sample in samples_list:
        graph_1_data, graph_2_data, list_of_matrix_1, list_of_matrix_2, idx1, idx2 = sample
        graph_data_list.append({
            'graph_1_data': graph_1_data,
            'graph_2_data': graph_2_data
        })
        other_data_list.append({
            'list_of_matrix_1': list_of_matrix_1,
            'list_of_matrix_2': list_of_matrix_2,
            'idx1': idx1,
            'idx2': idx2
        })
    
    # Save the list of graph data using torch.save
    torch.save(graph_data_list, filename + '_graphs.pth')
    
    # Save the other components using pickle
    with open(filename + '_others.pkl', 'wb') as f:
        pickle.dump(other_data_list, f)


def adjust_list(a, b):
    if len(a) >= b:
        # Return the first b elements
        return a[:b]
    else:
        # Repeat the last element until the list has size b
        return a + [a[-1]] * (b - len(a))


def save_labels(labels_list, filename):
    # Save the labels using pickle
    with open(filename, 'wb') as f:
        pickle.dump(labels_list, f)


# def save_benchmark_1():
#     args = config.parse()
#     # args.numL = 10
#     print(args.numL)
#     current_directory = os.getcwd()
#     data_name = 'benchmark1'
#     data_raw_dir = current_directory + f"/data/{data_name}/samples_raw"
#     data_pairs = current_directory + f"/data/{data_name}/processed_paried_{args.numL}"
#     data_labels = current_directory + f"/data/{data_name}/labels_{args.numL}"

#     graph_pairs = []
#     labels = []
#     sample = 0
#     for filename in sorted(os.listdir(data_raw_dir)):
#         if filename.endswith("-1"):
#             # print(sample)
#             g1_path = os.path.join(data_raw_dir, filename)
#             r1_path = g1_path + "_record.txt"
#             graph_1 = graph(g1_path, r1_path)
#             g2_path = find_graph_pair(g1_path)
#             r2_path = g2_path + "_record.txt"
#             graph_2 = graph(g2_path, r2_path)
            
#             numL = args.numL
#             idx1 = adjust_list(graph_1.max_nodes_idx, numL)
#             idx2 = adjust_list(graph_2.max_nodes_idx, numL)
#             one_sample = (graph_1.data, graph_2.data, graph_1.list_of_matrix, graph_2.list_of_matrix, idx1, idx2)
#             label = graph_1.label
#             print(sample, label)
#             graph_pairs.append(one_sample)
#             labels.append(label)
#             sample += 1
    
#     save_samples(graph_pairs, data_pairs)
#     save_labels(labels, data_labels)






def save_self_generated(data_name):
    
    current_directory = os.getcwd()
    # data_name = 'EXP'
    data_raw_dir = current_directory + f"/data/{data_name}/samples_raw"
    data_pairs = current_directory + f"/data/{data_name}/{data_name}_paried"
    data_labels = current_directory + f"/data/{data_name}/{data_name}_labels"

    self_generated_format_data = ['self_generated_data', 'self_generated_data_2', 'EXP', 'CEXP']

    graph_pairs = []
    labels = []
    sample = 0
    for filename in sorted(os.listdir(data_raw_dir)):
        if data_name in self_generated_format_data:
            if filename.endswith("-1.npy"):
                g1_path = os.path.join(data_raw_dir, filename)
                r1_path = g1_path.replace('.npy', '_record.txt')
                graph_1 = graph(g1_path, r1_path, dataset=data_name)
                if data_name == 'self_generated_data' or data_name == 'self_generated_data_2':
                    g2_path = g1_path.replace('-1.npy', '-2.npy')
                else:
                    g2_path = g1_path.replace('-1.npy', '-0.npy')
                if not os.path.exists(g2_path):
                    continue
                r2_path = g2_path.replace('.npy', '_record.txt')
                if not os.path.exists(r2_path):
                    continue
                graph_2 = graph(g2_path, r2_path, dataset=data_name)
                
                numL = 10
                idx1 = adjust_list(graph_1.max_nodes_idx, numL)
                # print(len(idx1))
                # sys.exit()
                idx2 = adjust_list(graph_2.max_nodes_idx, numL)
                # print(graph_1.data.x.shape)
                one_sample = (graph_1.data, graph_2.data, graph_1.list_of_matrix, graph_2.list_of_matrix, idx1, idx2)
                label = graph_1.label
                # label = 0
                print(sample, label)
                graph_pairs.append(one_sample)
                labels.append(label)
                sample += 1
        if data_name == 'benchmark1':
            if filename.endswith("-1"):
                g1_path = os.path.join(data_raw_dir, filename)
                r1_path = g1_path + "_record.txt"
                graph_1 = graph(g1_path, r1_path, dataset=data_name)
                g2_path = find_graph_pair(g1_path)
                r2_path = g2_path + "_record.txt"
                graph_2 = graph(g2_path, r2_path, dataset=data_name)
                
                numL = 10
                idx1 = adjust_list(graph_1.max_nodes_idx, numL)
                # print(len(idx1))
                # sys.exit()
                idx2 = adjust_list(graph_2.max_nodes_idx, numL)
                # print(graph_1.data.x.shape)
                one_sample = (graph_1.data, graph_2.data, graph_1.list_of_matrix, graph_2.list_of_matrix, idx1, idx2)
                label = graph_1.label
                # label = 0
                print(sample, label)
                graph_pairs.append(one_sample)
                labels.append(label)
                sample += 1
        
    
    save_samples(graph_pairs, data_pairs)
    save_labels(labels, data_labels)



data_name = 'EXP'
save_self_generated(data_name)

# save_benchmark_1()

