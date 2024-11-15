import torch
import config
from utils import graph, find_graph_pair, GraphPairDataset
from isgnn import SiameseNetwork, ISGNN, count_parameters, ExactModel
import os
from tqdm import tqdm
import numpy as np
import time
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import sys
import random
import csv
from torch_geometric.data import Batch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler

def pad_sparse_matrix(sparse_matrix, target_size):
    """
    Pads a sparse matrix to the target size by adding zero rows/columns.
    Args:
        sparse_matrix (torch.sparse_coo_tensor): Sparse matrix to pad.
        target_size (tuple): Desired size (rows, cols).
    """
    current_size = sparse_matrix.size()
    
    # If the sparse matrix is already the correct size, return it as is
    if current_size == target_size:
        return sparse_matrix
    
    indices = sparse_matrix.coalesce().indices()
    values = sparse_matrix.coalesce().values()

    # Adjust indices for padding
    pad_indices = indices[:, indices[0] < target_size[0]]
    pad_indices = pad_indices[:, indices[1] < target_size[1]]
    
    # Create a new sparse tensor with the padded size
    padded_matrix = torch.sparse_coo_tensor(pad_indices, values, size=target_size)
    return padded_matrix


class Custom_Loss(nn.Module):
    def __init__(self, model, lambda_reg=0.0001, margin=1000.0):

        super(Custom_Loss, self).__init__()
        self.model = model
        self.lambda_reg = lambda_reg
        self.margin = margin
        

    # def forward(self, score, y, predicted_idx1, predicted_idx2, true_idx1, true_idx2):
        
    def forward(self, score, y):

        ## score is L2 between two representations: 0 means identical; infinity means different
        
        loss_positive = y * torch.pow(score, 2)  # Loss when the objects are the same
        loss_negative = (1 - y) * torch.pow(torch.clamp(self.margin - score, min=0.0), 2)  # Loss when objects are not the same
        # loss_negative = (1 - y) * (1/score)

        # Combine the positive and negative losses
        # print(loss_positive, loss_negative)
        loss_1 = torch.mean(0.5 * (loss_positive + loss_negative))
        # print('loss_1:', loss_1)
        # sys.exit()
        loss_2 = 0

        # loss_2 = F.mse_loss(predicted_idx1.squeeze(-1), true_idx1)
        # loss_2 += F.mse_loss(predicted_idx2.squeeze(-1), true_idx2)
        # loss_2 = loss_2/2
        # print(loss_1.mean(), loss_2)
        # sys.exit()
        
        return loss_1.mean() + self.lambda_reg * loss_2




def evaluate_model(model, dataloader, criterion, device, purpose='valid'):
    model.eval()
    total_loss = 0.0
    accuracy = 0
    with torch.no_grad():
        for graph_pair, labels in dataloader:

            graph_1_data_tensor, graph_2_data_tensor, graph_1_matrix_tensor, graph_2_matrix_tensor, graph_1_nodes_tensor, graph_2_nodes_tensor = graph_pair
            graph_1_data_tensor = graph_1_data_tensor.to(device)
            graph_2_data_tensor = graph_2_data_tensor.to(device)
            # graph_1_nodes_tensor = graph_1_nodes_tensor.to(device)
            # graph_2_nodes_tensor = graph_2_nodes_tensor.to(device)
            labels = labels.to(device)
            score = model(graph_1_data_tensor, graph_2_data_tensor, labels)
            loss = criterion(score, labels)
            acc = eval_acc(labels, score, purpose)
            total_loss += loss.item()
            accuracy += acc

    return total_loss / (len(dataloader)), accuracy/(len(dataloader.dataset))


def eval_acc(y_true, score, purpose='train'):
    a = float('inf')
    predicted_labels = (score <= 10).float()
    correct_predictions = (predicted_labels == y_true)

    correct_num = correct_predictions.sum().item()
    # if purpose == 'valid':
    #     print(y_true)
    #     print(score)
    #     print(predicted_labels)
    #     print(correct_num)
    #     print('======================')

    return correct_num



def load_samples(filename):
    # Load the graph data
    graph_data_list = torch.load(filename + '_graphs.pth')
    
    # Load the other components
    with open(filename + '_others.pkl', 'rb') as f:
        other_data_list = pickle.load(f)
    
    # Combine the graph data and other components back into the original format
    samples_list = []
    for graph_data, other_data in zip(graph_data_list, other_data_list):
        sample = (
            graph_data['graph_1_data'], graph_data['graph_2_data'],
            other_data['list_of_matrix_1'], other_data['list_of_matrix_2'],
            other_data['idx1'], other_data['idx2']
        )
        samples_list.append(sample)
    
    return samples_list


def load_labels(filename):
    # Load the labels using pickle
    with open(filename, 'rb') as f:
        labels_list = pickle.load(f)
    return labels_list



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




def custom_collate_fn(batch):
    graph_1_data_list = []
    graph_2_data_list = []
    graph_1_matrix_list = []
    graph_2_matrix_list = []
    graph_1_nodes_list = []
    graph_2_nodes_list = []
    labels = []

    for (graph_1_data, graph_2_data, graph_1_matrices, graph_2_matrices, graph_1_nodes, graph_2_nodes), label in batch:
        graph_1_data_list.append(graph_1_data)
        graph_2_data_list.append(graph_2_data)
        # graph_1_matrix_list.append(graph_1_matrices)
        # graph_2_matrix_list.append(graph_2_matrices)

        # Convert node indices to tensors if they are lists
        if isinstance(graph_1_nodes, list):
            graph_1_nodes = torch.tensor(graph_1_nodes)
        if isinstance(graph_2_nodes, list):
            graph_2_nodes = torch.tensor(graph_2_nodes)

        graph_1_nodes_list.append(graph_1_nodes)
        graph_2_nodes_list.append(graph_2_nodes)
        labels.append(label)

    # Use PyTorch Geometric's Batch class to batch graph data
    graph_1_data_batch = Batch.from_data_list(graph_1_data_list)
    graph_2_data_batch = Batch.from_data_list(graph_2_data_list)

    # Batch node indices
    # print(graph_1_nodes_list)
    graph_1_nodes_batch = torch.stack(graph_1_nodes_list, dim=0).t()
    graph_2_nodes_batch = torch.stack(graph_2_nodes_list, dim=0).t()

    # Function to batch sparse matrices into a block-diagonal matrix
    def batch_sparse_matrices(matrices):
        device = matrices[0].device
        dtype = matrices[0].dtype
        total_size = sum([matrix.size(0) for matrix in matrices])
        cumulative_sizes = torch.tensor([0] + [matrix.size(0) for matrix in matrices[:-1]]).cumsum(0)

        all_indices = []
        all_values = []

        for matrix, offset in zip(matrices, cumulative_sizes):
            indices = matrix.coalesce().indices()
            values = matrix.coalesce().values()
            # Adjust indices by offset
            adjusted_indices = indices + offset.unsqueeze(0)
            all_indices.append(adjusted_indices)
            all_values.append(values)

        batched_indices = torch.cat(all_indices, dim=1)
        batched_values = torch.cat(all_values)
        size = (total_size, total_size)
        batched_matrix = torch.sparse_coo_tensor(batched_indices, batched_values, size=size, device=device, dtype=dtype)
        return batched_matrix

    # Batch graph_1_matrix_list
    # num_matrices_per_sample = len(graph_1_matrix_list[0])  # Assuming all samples have the same number of matrices
    # graph_1_matrix_batch = []
    # for i in range(num_matrices_per_sample):
    #     matrices_to_batch = [sample[i] for sample in graph_1_matrix_list]
    #     batched_matrix = batch_sparse_matrices(matrices_to_batch)
    #     graph_1_matrix_batch.append(batched_matrix)

    # # Batch graph_2_matrix_list
    # num_matrices_per_sample = len(graph_2_matrix_list[0])
    # graph_2_matrix_batch = []
    # for i in range(num_matrices_per_sample):
    #     matrices_to_batch = [sample[i] for sample in graph_2_matrix_list]
    #     batched_matrix = batch_sparse_matrices(matrices_to_batch)
    #     graph_2_matrix_batch.append(batched_matrix)

    labels_batch = torch.stack(labels)

    return (graph_1_data_batch, graph_2_data_batch, graph_1_matrix_list, graph_2_matrix_list,
            graph_1_nodes_batch, graph_2_nodes_batch), labels_batch


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        print("CustomDataLoader is being initialized")
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn, **kwargs)



def main():
    
    args = config.parse()
    current_directory = os.getcwd()


    data_name = args.dname
    print(f'Load Dataset: {data_name}')

    # if data_name == 'benchmark1':
        # data_raw_dir = current_directory + f"/data/{data_name}/samples_raw"
    current_time = time.time()
    local_time = time.localtime(current_time)
    formatted_time = time.strftime("%b_%d_%H%M%S", local_time)
    
    log_file = current_directory + f"/results/log/{data_name}_{formatted_time}.csv"
    results_file = current_directory + f"/results/final/{data_name}_{formatted_time}.csv"

    data_pairs = current_directory + f"/data/{data_name}/{data_name}_paried"
    data_labels = current_directory + f"/data/{data_name}/{data_name}_labels"


    graph_pairs = load_samples(data_pairs)
    labels = load_labels(data_labels)
    num_samples = len(graph_pairs)
    print(f'Num_samples: {num_samples}')
    dataset = GraphPairDataset(graph_pairs, labels)
    prop = args.prop

    
    print(f'Prepare Model')
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                            if torch.cuda.is_available() else 'cpu')
    # isgnn = ISGNN(args.output_dim, args.GIN_num_layers, args.GIN_hidden_dim, args.numL, args.numK)
    # isgnn.to(device)
    
    # model = SiameseNetwork(isgnn, device)
    model = SiameseNetwork(device)

    # model = ExactModel()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_params = count_parameters(model)
    # for name, param in model.named_parameters():
    #     print(name, param.shape)
    # sys.exit()
    Loss = Custom_Loss(model, args.reg_lambda)
    
    train_size = int(args.train_prop * num_samples)
    valid_size = int(args.valid_prop * num_samples)
    test_size = num_samples - train_size - valid_size

    runtime_list = []
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        title = ['learning rate', 'GIN_num_layers', 'GIN_hidden_dim', 'train_patience', 'numL', 'numK']
        writer.writerow(title)

        train_hyper_data = [args.lr, args.GIN_num_layers, args.GIN_hidden_dim, args.train_patience, args.numL, args.numK]
        writer.writerow(train_hyper_data)
        seed = [random.randint(1, 10)+i for i in range(args.runs)]
        writer.writerow(seed)

        title_2 = ['run', 'epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'test_loss', 'test_acc', 'epoch_duration', 'run_duration']
        writer.writerow(title_2)
        print(f'Start Training')
        
        for run in tqdm(range(args.runs)):
            start_time = time.time()

            set_seed(seed[run])  # Set the seed before running the training
            # Split the dataset
            train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
            
            # model.reset_parameters()
            
            best_valid_loss = float('inf')
            epochs_no_improve = 0  # Counter for early stopping
            early_stop = False

            for epoch in range(args.epochs):
                if early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
                start_epoch = time.time()
                training_loss = 0.0
                training_acc_init = 0
                model.train()
                
                for graph_pair, label in train_loader:
                    # print(f'lable: {label}')
                    

                    graph_1_data_tensor, graph_2_data_tensor, graph_1_matrix_tensor, graph_2_matrix_tensor, graph_1_nodes_tensor, graph_2_nodes_tensor = graph_pair
                    graph_1_data_tensor = graph_1_data_tensor.to(device)
                    graph_2_data_tensor = graph_2_data_tensor.to(device)

                    # graph_1_nodes_tensor = graph_1_nodes_tensor.to(device)
                    # graph_2_nodes_tensor = graph_2_nodes_tensor.to(device)

                    optimizer.zero_grad()
                    
                    # score, cl1, cl2, idx1, idx2 = model(graph_1_data_tensor, graph_2_data_tensor)
                    score = model(graph_1_data_tensor, graph_2_data_tensor, label)
                    # sys.exit()

                    label = label.to(device)
                    loss = Loss(score, label)
                    # loss = score

                    train_acc = eval_acc(label, score)
                    training_acc_init += train_acc
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    training_loss += loss.item()
                train_time = time.time() - start_epoch 
                train_loss = training_loss / len(train_loader)
                # print(training_acc_init)

                train_acc_ = training_acc_init / len(train_loader.dataset)
                valid_loss, valid_acc = evaluate_model(model, valid_loader, Loss, device, purpose='valid')
                test_loss, test_acc = evaluate_model(model, test_loader, Loss, device, purpose='test')
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    epochs_no_improve = 0 
                else:
                    epochs_no_improve += 1 

                if epoch % args.display_step == 0 and args.display_step > 0:
                    print(f'------'
                        f'Epoch: {epoch:02d}, '
                        f'Train Loss: {train_loss:.4f}, '
                        f'Train acc: {train_acc_:.4f}, '
                        f'Valid Loss: {valid_loss:.4f}, '
                        f'Valid acc: {valid_acc:.4f}, '
                        f'Test Loss: {test_loss:.4f}, '
                        f'Test acc: {test_acc:.4f}, '
                        f'Train Time One Epoch: {train_time} s')
                
                data = [run, epoch, train_loss, train_acc_, valid_loss, valid_acc, test_loss, test_acc, train_time, '_']

                writer.writerow(data)

                # Early stopping: stop training if validation loss increases several times
                if epochs_no_improve >= args.train_patience:
                    print(f"Validation loss increased for {args.train_patience} consecutive epochs. Stopping training.")
                    early_stop = True


            
            one_run_time = time.time() - start_time

            print(f'Results for Run {run}, '
                f'Train Loss: {train_loss:.4f}, '
                f'Train acc: {train_acc_:.4f}, '
                f'Valid Loss: {valid_loss:.4f}, '
                f'Valid acc: {valid_acc:.4f}, '
                f'Test Loss: {test_loss:.4f}, '
                f'Test acc: {test_acc:.4f}, '
                f'One Run Time: {one_run_time} s')
            data = [run, '_', train_loss, train_acc_, valid_loss, valid_acc, test_loss, test_acc, '_', one_run_time]

            writer.writerow(data)
            
    print('All done! Exit python code')


if __name__ == "__main__":
    # log = 
    main()


