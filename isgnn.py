import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from torch_geometric.nn import GINConv, global_add_pool
import math
import torch.nn.init as init
from torch_geometric.utils import softmax
import matplotlib.pyplot as plt
import networkx as nx
import sys
from layers import GINModel, InfoCollect, IndiiLayer, TrainableHardThreshold, DynamicTrainableStepFunctionF5, DynamicTrainableStepFunctionF5_batch, LocalInfo
from torch_geometric.data import Data
import time
# import torch_scatter
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import add_self_loops, to_networkx
from utils import check_iso, draw_two_graphs
from collections import Counter



class TreeNode(object):
    def __init__(self, partition):
        self.partition = partition
        self.score = None
        self.parent = None
        self.child = []



class CustomSignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, epsilon):
        ctx.save_for_backward(input)
        ctx.k = k
        ctx.epsilon = epsilon
        return torch.tanh(k * input - epsilon)
    

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        k = ctx.k
        epsilon = ctx.epsilon
        tanh_output = torch.tanh(k * input - epsilon)
        grad_input = grad_output * k * (1 - tanh_output ** 2)
        return grad_input, None, None 


class ExactModel(nn.Module):
    def __init__(self, k=1000, epsilon=5, prop=1):
        super(ExactModel, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.relu = nn.ReLU()
        self.f4 = TrainableHardThreshold()
        self.f5 = DynamicTrainableStepFunctionF5()
        self.prop = prop


    def is_discrete(self, partition):
        # Input Validation
        if not isinstance(partition, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if partition.dim() != 1:
            raise ValueError("Input tensor must be 1-dimensional.")
        
        # Handle Empty Tensor: Consider empty tensor as having all unique elements
        if partition.numel() == 0:
            return True
        # Extract Unique Elements
        unique_elements = torch.unique(partition)
        # Compare Number of Unique Elements with Total Elements
        return unique_elements.numel() == partition.numel()
    
    def infoco1(self, data: Data, p: torch.Tensor, device) -> torch.Tensor:
        """
        Computes the local information S for each node in a single directed graph,
        including self-loops.
        
        Args:
            data (torch_geometric.data.Data): Graph data containing edge_index and num_nodes.
            p (torch.Tensor): Partition vector of shape (num_nodes,), with elements in {1, ..., N_g}.
        
        Returns:
            torch.Tensor: Local information vector S of shape (num_nodes,).
        """
        # Number of nodes in the graph
        # total_nodes = data.num_nodes
        
        # # Validate partition vector
        # assert p.dim() == 1 and p.size(0) == total_nodes, \
        #     "Partition vector p must be a 1D tensor with length equal to the number of nodes."
        # assert torch.all((p >= 1) & (p <= total_nodes)), \
        #     "All elements of partition vector p must be in the range {1, ..., N_g}."
        # p = p.float()  # Convert to float for computation
        
        # # Compute N_g (number of nodes)
        # N_g = float(total_nodes)
        
        # # Add self-loops to the edge_index
        # edge_index_with_loops, _ = add_self_loops(data.edge_index, num_nodes=total_nodes)
        

        # # Compute N_g^{p_j} for each node j
        # # N_power_p = torch.pow(torch.full((total_nodes,), N_g**2, dtype=p.dtype), p).to(device)  # Shape: (num_nodes,)
        # N_power_p = p

        # # Extract source and target nodes from edge_index_with_loops
        # # edge_index has shape [2, num_edges], where edge_index[0] are source nodes (i) and edge_index[1] are target nodes (j)
        # source_nodes = edge_index_with_loops[0]  # Shape: (num_edges,)
        # target_nodes = edge_index_with_loops[1]  # Shape: (num_edges,)
        
        # # Retrieve p_i for each edge i->j
        # p_i = p[source_nodes]  # Shape: (num_edges,)
        
        # # Retrieve N_g^{p_j} for each edge i->j based on target node j
        # N_g_p_j = N_power_p[target_nodes]  # Shape: (num_edges,)
        
        # # Compute p_i * N_g^{p_j} for each edge i->j
        # contributions = p_i * N_g_p_j  # Shape: (num_edges,)
        # # print(torch.log(contributions))
        
        # # Aggregate contributions to each target node j
        # S = scatter_add(contributions, target_nodes, dim=0, dim_size=total_nodes)  # Shape: (num_nodes,)

        total_nodes = data.num_nodes
        
        # Validate partition vector
        assert p.dim() == 1 and p.size(0) == total_nodes, \
            "Partition vector p must be a 1D tensor with length equal to the number of nodes."
        assert torch.all((p >= 1) & (p <= total_nodes)), \
            f"All elements of partition vector p must be in the range (1, ..., N_g), currently {p}."
        p = p.float()  # Convert to float for computation
        
        # Compute N_g (number of nodes)
        N_g = float(total_nodes)
        
        # Add self-loops to the edge_index
        edge_index_with_loops, _ = add_self_loops(data.edge_index, num_nodes=total_nodes)
        
        # Compute log(N_g^p) = p * log(N_g)
        N_power_p_log = p * torch.log(torch.tensor(N_g, device=device))
        
        # Extract source and target nodes from edge_index_with_loops
        source_nodes = edge_index_with_loops[0]  # Shape: (num_edges,)
        target_nodes = edge_index_with_loops[1]  # Shape: (num_edges,)
        
        # Retrieve p_i and p_j for each edge i->j
        p_i = p[source_nodes]  # Shape: (num_edges,)
        N_g_p_j_log = N_power_p_log[target_nodes]  # Shape: (num_edges,)
        
        # Compute contributions in log-space
        contributions_log = torch.log(p_i) + N_g_p_j_log  # Shape: (num_edges,)
        
        # Compute max per target node to use in log-sum-exp
        max_per_target, _ = scatter_max(contributions_log, target_nodes, dim=0, dim_size=total_nodes)
        
        # Handle nodes with no incoming edges
        non_zero_mask = max_per_target != float('-inf')
        
        # Normalize contributions for numerical stability
        exp_contributions = torch.exp(contributions_log - max_per_target[target_nodes])
        
        # Sum exponentials per target node
        sum_exp = scatter_add(exp_contributions, target_nodes, dim=0, dim_size=total_nodes)
        
        # Compute log-sum-exp per node
        S_log = torch.full((total_nodes,), float('-inf'), device=device)
        S_log[non_zero_mask] = max_per_target[non_zero_mask] + torch.log(sum_exp[non_zero_mask])


        S = S_log + torch.dot(data.x.squeeze(), p)
        
        return S

    def infoco(self, data: Data, p: torch.Tensor, device) -> torch.Tensor:
        """
        Computes the local information S for each node in a single directed graph,
        including self-loops, while avoiding numerical overflow.
        
        Args:
            data (torch_geometric.data.Data): Graph data containing edge_index and num_nodes.
            p (torch.Tensor): Partition vector of shape (num_nodes,), with elements in {1, ..., N_g}.
        
        Returns:
            torch.Tensor: Local information vector S_log of shape (num_nodes,).
        """
        total_nodes = data.num_nodes
        
        # Validate partition vector
        assert p.dim() == 1 and p.size(0) == total_nodes, \
            "Partition vector p must be a 1D tensor with length equal to the number of nodes."
        assert torch.all((p >= 1) & (p <= total_nodes)), \
            f"All elements of partition vector p must be in the range (1, ..., N_g), currently {p}."
        p = p.float()  # Convert to float for computation
        
        # Compute N_g (number of nodes)
        N_g = float(total_nodes)
        
        # Add self-loops to the edge_index
        edge_index_with_loops, _ = add_self_loops(data.edge_index, num_nodes=total_nodes)
        
        # Compute log(N_g^p) = p * log(N_g)
        N_power_p_log = p * torch.log(torch.tensor(N_g, device=device))
        
        # Extract source and target nodes from edge_index_with_loops
        source_nodes = edge_index_with_loops[0]  # Shape: (num_edges,)
        target_nodes = edge_index_with_loops[1]  # Shape: (num_edges,)
        
        # Retrieve p_i and p_j for each edge i->j
        p_i = p[source_nodes]  # Shape: (num_edges,)
        N_g_p_j_log = N_power_p_log[target_nodes]  # Shape: (num_edges,)
        
        # Compute contributions in log-space
        contributions_log = torch.log(p_i) + N_g_p_j_log  # Shape: (num_edges,)
        # print(contributions_log.device)

        # print(f"Device of contributions_log: {contributions_log.device}")
        # print(f"Device of target_nodes: {target_nodes.device}")
        # print(f"Expected device: {device}")
        # CUDA_LAUNCH_BLOCKING=1

        
        
        max_per_target, _ = scatter_max(contributions_log, target_nodes, dim=0, dim_size=total_nodes)
        
        # Handle nodes with no incoming edges
        non_zero_mask = max_per_target != float('-inf')
        
        # Normalize contributions for numerical stability
        exp_contributions = torch.exp(contributions_log - max_per_target[target_nodes])
        
        # Sum exponentials per target node
        sum_exp = scatter_add(exp_contributions, target_nodes, dim=0, dim_size=total_nodes)
        
        # Compute log-sum-exp per node
        S_log = torch.full((total_nodes,), float('-inf'), device=device)
        S_log[non_zero_mask] = max_per_target[non_zero_mask] + torch.log(sum_exp[non_zero_mask])
        
        return S_log

    def refine(self, info, num_nodes, data, device):
        # print(device)
        fixed = False
        prev = info
        time = 0
        while not fixed:
            time += 1
            # print(f'refined times {time}')
            if time == 1:
                X = info.view(-1, 1)
            else:
                X = self.infoco(data, prev, device=device).view(-1, 1)
            # print(X)
            D = X - X.T
            D_1 = CustomSignFunction.apply(D, self.k, self.epsilon)
            # print('=========================')
            # print(D)

            D = D_1.sum(dim=1, keepdim=True)

            new_p = self.f5(D, num_nodes).squeeze()
            # print(new_p.shape)
            # print(D)
            # print('---------------------')
            # print(new_p)
            # sys.exit()
            if torch.equal(new_p, prev):
                fixed = True
            else:
                # print(new_p)
                # print(prev)
                # sys.exit()
                prev = new_p
            # if time == 3:
            #     sys.exit()
        new_p = new_p.squeeze()
        return new_p

    def inid(self, partition, idx, length, num_nodes):
        if not isinstance(partition, torch.Tensor):
            raise TypeError("A must be a torch.Tensor.")
        if partition.dim() != 1:
            raise ValueError("A must be a 1D tensor.")
        n = partition.size(0)
        if not (0 <= idx < n):
            raise IndexError(f"Index i must be in the range [0, {n-1}].")
        
        # Clone A to create B
        B = partition.clone()
        # Increment the i-th element
        
        

        if partition[idx] != num_nodes:
            B[idx] = partition[idx] + length - 1
        else:
            B[idx] = partition[idx] - (length - 1)
        return B

    def cl_form(self, leaves):
        leaves_converted = {key: [tensor.tolist() for tensor in value] for key, value in leaves.items()}
        # for key, value in leaves_converted.items():
        #     print(value)
        # sys.exit()
        max_cl = max(leaves_converted, key=lambda k:leaves_converted[k])
        keys_list = list(leaves_converted.keys())

        # Find the index of the key in the keys list
        max_cl_index = keys_list.index(max_cl)
        print('decision', max_cl_index, len(keys_list))
        # print(max_cl)
        # lst = max_cl.tolist()
        # for idx, ele in enumerate(lst):
        #     print(int(ele), idx)
        return max_cl

    def target_cell(self, partition):
        """
        Returns a list of indices where the values in tensor 'partition' have the highest occurrence.
        If there is a tie in max count, returns the indices corresponding to the maximum value among the modes.

        Args:
            partition (torch.Tensor): Input tensor of shape (n,).

        Returns:
            List[int]: List of indices corresponding to the mode with the maximum value.
        """
        if not isinstance(partition, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if partition.dim() != 1:
            raise ValueError("Input tensor must be 1-dimensional.")
        if partition.numel() == 0:
            return []

        unique_values, counts = torch.unique(partition, return_counts=True)
        # print(unique_values, counts)
        max_count = counts.max()
        mode_values = unique_values[counts == max_count]

        # Find the mode value with the maximum value among modes
        max_mode_value = mode_values.max()

        # Get indices where partition equals max_mode_value
        mask = (partition == max_mode_value)
        indices = torch.nonzero(mask, as_tuple=False).squeeze().tolist()

        # Ensure indices is a list
        if isinstance(indices, int):
            indices = [indices]
        elif not isinstance(indices, list):
            indices = indices.tolist()
        # print(partition)
        return indices

    def torch_lexsort(self, keys):
        """
        Perform lexicographical sorting on a list of 1D tensors.

        Parameters:
        keys : list of torch.Tensor
            A list of 1D tensors representing the sorting keys.

        Returns:
        indices : torch.Tensor
            A tensor of indices that sorts the data lexicographically.
        """
        assert len(keys) > 0, "At least one key is required for lexsort."
        # Start with indices ranging from 0 to N-1
        indices = torch.arange(keys[0].size(0))
        # Perform sorting starting from the last key
        for key in reversed(keys):
            _, indices = torch.sort(key[indices], stable=True)
        return indices

    def compute_essential_info(self, a, edge_index, device):
        """
        Given a partition 'a' and 'edge_index', compute the enhanced essential information
        as a PyTorch tensor.

        Parameters:
        a : torch.Tensor
            A tensor of shape (n,), where a[i] is the cell ID of node i.
        edge_index : torch.Tensor
            A tensor of shape (2, m), where edge_index[:, k] represents an edge
            from edge_index[0, k] to edge_index[1, k].
        device : torch.device
            The device on which to perform computations.

        Returns:
        essential_info : torch.Tensor
            A 1D tensor containing the enhanced essential information.
        """
        if device is None:
            device = a.device

        # Step 1: Get unique cell IDs and their sizes
        unique_cells, counts = torch.unique(a, sorted=False, return_counts=True)

        # Step 2: Sort cells by counts (descending) and cell IDs (ascending) for determinism
        counts_neg = -counts  # For descending sort
        sorting_keys = [counts_neg, unique_cells]
        sorted_indices = self.torch_lexsort(sorting_keys)

        # Apply the sorted indices to counts and unique_cells
        sorted_counts = counts[sorted_indices]
        sorted_cell_ids = unique_cells[sorted_indices]

        num_cells = len(sorted_cell_ids)

        # Mapping from cell ID to index in sorted list
        cell_id_to_index = {int(cell_id.item()): idx for idx, cell_id in enumerate(sorted_cell_ids)}

        # Step 3: Compute edge counts between cells
        edge_counts = torch.zeros((num_cells, num_cells), dtype=torch.int64, device=device)
        sources = edge_index[0]
        targets = edge_index[1]

        for u, v in zip(sources.tolist(), targets.tolist()):
            cell_u = a[u].item()
            cell_v = a[v].item()
            idx_u = cell_id_to_index[cell_u]
            idx_v = cell_id_to_index[cell_v]
            edge_counts[idx_u, idx_v] += 1

        # Step 4: Collect node degrees within cells and internal/external edge counts
        degrees = torch.zeros(a.size(0), dtype=torch.int64, device=device)
        for u, v in zip(sources.tolist(), targets.tolist()):
            degrees[u] += 1  # For undirected graphs; adjust if directed

        cell_degrees = []
        internal_edge_counts = []
        external_edge_counts = []
        for idx, cell_id in enumerate(sorted_cell_ids):
            node_indices = (a == cell_id).nonzero(as_tuple=True)[0].to(device)
            cell_node_degrees = degrees[node_indices]
            sorted_degrees = torch.sort(cell_node_degrees)[0]
            cell_degrees.append(sorted_degrees)

            # Compute internal and external edges
            internal_edges = 0
            external_edges = 0
            node_set = set(node_indices.tolist())
            for u in node_indices.tolist():
                neighbors = edge_index[1][edge_index[0] == u].tolist()
                for v in neighbors:
                    if v in node_set:
                        internal_edges += 1
                    else:
                        external_edges += 1
            # Since each internal edge is counted twice (once for each node), divide by 2
            internal_edges = internal_edges // 2
            internal_edge_counts.append(torch.tensor([internal_edges], dtype=torch.int64, device=device))
            external_edge_counts.append(torch.tensor([external_edges], dtype=torch.int64, device=device))

        # Step 5: Flatten the edge counts matrix in a deterministic order
        indices_upper = torch.triu_indices(num_cells, num_cells)
        edge_counts_flat = edge_counts[indices_upper[0], indices_upper[1]]

        # Step 6: Combine all information into a single tensor
        essential_info_list = [sorted_counts, edge_counts_flat]
        for idx in range(num_cells):
            essential_info_list.append(cell_degrees[idx])
            essential_info_list.append(internal_edge_counts[idx])
            essential_info_list.append(external_edge_counts[idx])

        # Concatenate all tensors into a single 1D tensor
        essential_info = torch.cat(essential_info_list)

        return essential_info

    def compare_sequences(self, sequences, edge_index, device):
        """
        Compare multiple sequences of partitions and return the one with the largest
        essential information based on lexicographic order.

        Parameters:
        sequences : list of list of torch.Tensor
            A list where each element is a sequence (list) of partitions 'a'.
        edge_index : torch.Tensor
            The edge_index of the graph.

        Returns:
        largest_sequence : list of torch.Tensor
            The sequence with the largest essential information.
        """
        sequence_reps = []
        for idx, seq in enumerate(sequences):
            seq_rep = []
            for a in seq:
                essential_info = self.compute_essential_info(a, edge_index, device)
                # Convert essential_info to a tuple for lexicographic comparison
                essential_info_tuple = tuple(essential_info.tolist())
                seq_rep.append(essential_info_tuple)
            # Convert sequence representation to a tuple
            seq_rep_tuple = tuple(seq_rep)
            sequence_reps.append((seq_rep_tuple, idx))

        # Sort sequences lexicographically in reverse to get the largest first
        sequence_reps.sort(reverse=True)

        # Get the index of the largest sequence
        largest_idx = sequence_reps[0][1]
        # largest_sequence = sequences[largest_idx]
        return largest_idx

    def refine_partition(self, a, edge_index):
        """
        Refines the partition 'a' of a graph with edges given by 'edge_index' using tensor operations.

        Parameters:
        a : torch.Tensor
            A tensor of shape (n,), where a[i] is the cell ID of node i.
        edge_index : torch.Tensor
            A tensor of shape (2, num_edges), where edge_index[:, k] represents an edge
            from edge_index[0, k] to edge_index[1, k].

        Returns:
        a_refined : torch.Tensor
            The refined partition tensor of shape (n,).
        """
        device = a.device  # Use the same device as 'a'
        n = a.size(0)
        # print('here')
        # print(device)
        # Initialize π and α
        # π is represented by 'a' itself
        # α is initialized with the first cell ID
        unique_cells = torch.unique(a)
        # We can initialize α with the first cell ID
        alpha = set([a[0].item()])

        # We'll also need to assign new cell IDs during refinement
        # Initialize next_cell_id to the maximum cell ID in 'a' plus one
        next_cell_id = a.max().item() + 1

        # Define a function to check if π is discrete
        def is_discrete(a):
            # π is discrete if every node is in its own cell
            unique_cells, counts = torch.unique(a, return_counts=True)
            return torch.all(counts == 1)

        # Refinement Loop
        while alpha and not is_discrete(a):
            # Remove some element W from α
            W_cell_id = alpha.pop()
            # Get the nodes in cell W
            W_nodes = (a == W_cell_id).nonzero(as_tuple=True)[0].view(-1)  # Ensure 1D tensor

            # For each cell X in π
            cell_ids = torch.unique(a)
            for X_cell_id in cell_ids:
                # Skip if X is empty or same as W
                if X_cell_id == W_cell_id:
                    continue

                # Get the nodes in cell X
                X_nodes = (a == X_cell_id).nonzero(as_tuple=True)[0].view(-1)  # Ensure 1D tensor

                # Skip if X is empty
                if X_nodes.numel() == 0:
                    continue

                # Compute the number of edges from each node in X to nodes in W
                # Create masks for edges where source node is in X and target node is in W
                src_in_X = torch.isin(edge_index[0], X_nodes)
                tgt_in_W = torch.isin(edge_index[1], W_nodes)
                # Edges from X to W
                edges_from_X_to_W = src_in_X & tgt_in_W
                # Get the source nodes of edges from X to W
                edge_src_nodes = edge_index[0][edges_from_X_to_W]

                # Map node indices in X to positions 0..len(X_nodes)-1
                node_positions_in_X = torch.arange(X_nodes.size(0), device=device)
                node_index_to_position = torch.full((n,), -1, dtype=torch.long, device=device)
                node_index_to_position[X_nodes] = node_positions_in_X

                # Get positions of edge source nodes in X
                positions_in_X = node_index_to_position[edge_src_nodes]

                # Increment counts for each node in X based on edges to W
                counts_X = torch.zeros(X_nodes.size(0), dtype=torch.long, device=device)
                counts_X.scatter_add_(0, positions_in_X, torch.ones_like(positions_in_X, dtype=counts_X.dtype))

                # Group nodes in X based on counts_X
                counts_unique, counts_inv = torch.unique(counts_X, return_inverse=True)

                # Assign new cell IDs to the fragments
                num_fragments = counts_unique.size(0)
                new_cell_ids = torch.arange(num_fragments, device=device) + next_cell_id
                next_cell_id += num_fragments

                # Update 'a' for nodes in X
                a[X_nodes] = new_cell_ids[counts_inv].unsqueeze(1)

                # Update α
                if X_cell_id in alpha:
                    # Replace X_cell_id with new_cell_ids in α
                    alpha.remove(X_cell_id)
                    alpha.update(new_cell_ids.tolist())
                else:
                    # Add all but one of the largest fragments to α
                    # Compute sizes of fragments
                    fragment_sizes = torch.zeros(num_fragments, dtype=torch.long, device=device)
                    fragment_sizes.scatter_add_(0, counts_inv, torch.ones_like(counts_inv, dtype=fragment_sizes.dtype))
                    # Find indices of fragments sorted by size (descending)
                    sorted_indices = torch.argsort(fragment_sizes, descending=True)
                    # Exclude one of the largest fragments from α
                    largest_fragment_cell_id = new_cell_ids[sorted_indices[0]].item()
                    fragments_to_add = set(new_cell_ids.tolist())
                    fragments_to_add.remove(largest_fragment_cell_id)
                    alpha.update(fragments_to_add)

        # Return the refined partition
        return a

    def forward(self, data, device):
        # print('start===================================================================================')
        leaves = {}
        # print(f'number of nodes: {data.x.shape[0]}')
        # print(data.x.device)
        # print(device)
        num_nodes = data.x.shape[0]
        current_layer = []
        next_layer = []
        # print(data.x)
        # print(data.x.shape)
        # p_0 = self.refine(data.x, num_nodes, data, device)

        p_0 = self.refine_partition(data.x, data.edge_index)
        # print('here')
        
        
        root = TreeNode(p_0)
        root.score = [self.compute_essential_info(p_0, data.edge_index, device)]
        # root.score.append(torch.sum(self.infoco1(data, p_0, device)).item())
        # root.score.append(p_0)
        # print(p_0)
        # sys.exit()

        current_layer.append(root)
        level = 0

        while current_layer:
            # print('level', level)
            level += 1
            node_in_current_layer = 1
            # print(len(current_layer))
            base_score = current_layer[0].score

            for curr_treeNode in current_layer:
                curr_score = curr_treeNode.score
                node_in_current_layer += 1
                if self.is_discrete(curr_treeNode.partition.squeeze()):
                    leaves[curr_treeNode.partition] = curr_treeNode.score

                elif curr_score != base_score:
                    # print('here2')
                    continue

                elif curr_score < base_score:
                    # print('here3')
                    continue
                
                else:
                    base_score = curr_score
                    target_cell = self.target_cell(curr_treeNode.partition.squeeze())
                    # print(target_cell)
                    lst = []
                    for idx, val in enumerate(curr_treeNode.partition.tolist()):
                        lst.append((idx, val))
                    # print(lst)

                    # print(target_cell)
                    
                    for idx in target_cell:
                        temp_partition = self.inid(curr_treeNode.partition.squeeze(), idx, len(target_cell), num_nodes)
                        # new_info = self.infoco(data, temp_partition, device)
                        # new_partition = self.refine(new_info, num_nodes, data, device)
                        new_partition = self.refine_partition(temp_partition.unsqueeze(1), data.edge_index)
                        one_child = TreeNode(new_partition)
                        one_child.parent = curr_treeNode
                        temp_score = curr_treeNode.score
                        temp_score.append(self.compute_essential_info(new_partition, data.edge_index, device))
                        one_child.score = temp_score
                        curr_treeNode.child.append(one_child)
                        next_layer.append(one_child)

            print(len(next_layer))
            # print('======================================')
            if len(next_layer) == 1:
                current_layer = next_layer
            else:
                leng = math.ceil((len(next_layer)*self.prop))
                # print(leng)
                current_layer = next_layer[:leng]
            next_layer = []
        # print(len(leaves))
        # print(i for i in leaves.keys())
        # print(leaves)
        # sequences = list(leaves.values())
        # idx = self.compare_sequences(sequences, data.edge_index, device)
        cl_form = self.cl_form(leaves)

        # print('final---------')
        lst = []
        for idx, val in enumerate(cl_form.tolist()):
            lst.append((idx, val))
        # print(lst)




        # print(f'choose leaf {idx}')
        # cl_form = list(leaves.keys())[idx]
         
        return cl_form, leaves

    



class ISGNN(nn.Module):
    def __init__(self, output_dim, GIN_num_layers, GIN_hidden_dim, num_L, num_K, aggr='add', use_bias = False, k=1000, epsilon=5):
        super(ISGNN, self).__init__()
        # self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_L = num_L
        self.num_K = num_K
        self.use_bias = use_bias
        self.GIN_layers = GIN_num_layers

        self.InfoCollect_1 = InfoCollect(1, 8)
        self.InfoCollect_2 = InfoCollect(8, 8)
        self.InfoCollect_3 = InfoCollect(8, 1)
        
        self.Indi = IndiiLayer(1, 1)
        self.relu = nn.ReLU()
        self.f4 = TrainableHardThreshold()
        self.f5 = DynamicTrainableStepFunctionF5_batch()
        self.gin = GINModel(1, GIN_hidden_dim, self.output_dim, self.GIN_layers)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.k = k
        self.epsilon = epsilon

    # def reset_parameters(self):
    #     init.kaiming_uniform_(self.weight)
    #     # for layer in self.layers:
    #     #     layer.reset_parameters()
    #     if self.use_bias:
    #         init.zeros_(self.bias)
    
    def soft_argmax(self, input_tensor, temperature=1.0):
        """
        Soft approximation of argmax using softmax with temperature scaling.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (n, 1).
            temperature (float): Temperature parameter to control sharpness.
            
        Returns:
            torch.Tensor: Tensor of same shape as input with soft argmax approximation.
        """
        softmax_output = F.softmax(input_tensor / temperature, dim=0)
        return softmax_output



    def forward(self, data, device):
        time_0 = time.time()
        # print(device)
        # print(data.device)
        batch_size = data.num_graphs
        batch = data.batch.to(device)
        num_nodes_per_graph = torch.bincount(batch)
        cumulative_lengths = torch.cat([torch.tensor([0], device=batch.device), torch.cumsum(num_nodes_per_graph, dim=0)])

        node_indices = []
        y = torch.tensor([1, 1], device=device, dtype=torch.float32).unsqueeze(1).repeat(data.num_nodes, 1)  # Initialize y
        time_1 = time.time()


        # X = data.x.view(-1, 1)
        # D = X - X.T
        

        # D_1 = CustomSignFunction.apply(D, self.k, self.epsilon)
        # # print(D_1)
        # # sys.exit()
        # # time_2_2 = time.time()

        # D = D_1.sum(dim=1, keepdim=True)
        # print(D)
        # time_2_3 = time.time()
        
        # y = self.f5(D, num_nodes_per_graph)
        # # print(data.x.shape[0])
        # print(y)
        # sys.exit()
        
        for i in range(self.num_L):
            # Refinement
            time_2 = time.time()
            input_features = y if i != 0 else data.x
            collect_info = self.InfoCollect_1(input_features, data.edge_index)
            collect_info = self.InfoCollect_2(collect_info, data.edge_index)
            collect_info = self.InfoCollect_3(collect_info, data.edge_index).t()
            
            # print(collect_info)


            X = collect_info.view(-1, 1)
            D = X - X.T
            

            D_1 = CustomSignFunction.apply(D, self.k, self.epsilon)
            # print(D_1)
            # sys.exit()
            time_2_2 = time.time()

            D = D_1.sum(dim=1, keepdim=True)
            time_2_3 = time.time()
            
            y = self.f5(D, num_nodes_per_graph)
            

            time_3 = time.time()
            t_1 = time_3 - time_2
            # print('time')
            # print((time_2_2-time_2), (time_2_3-time_2_2), (time_3-time_2_3))
            # print('Duration')
            
            # print((time_2_2-time_2)/t_1, (time_2_3-time_2_2)/t_1, (time_3-time_2_3)/t_1)
            # print(y.shape)

            # print((time_2_2-time_2)/t_1, (time_3-time_2_2)/t_1)


            # Individualization - Collect invariant info
            out = self.Indi(y, data.edge_index)
            relu_out = F.relu(out).squeeze(1)
            time_4 = time.time()
            t_2 = time_4 - time_3


            # Individualization - Find node index with maximum value
            per_graph_node_indices = torch.arange(data.num_nodes, device=device) - cumulative_lengths[batch]
            p = softmax(relu_out / 0.1, batch)
            expected_indices = scatter_add(p * per_graph_node_indices.float(), batch, dim=0)
            self.argmax_out = expected_indices.unsqueeze(1)
            node_indices.append(self.argmax_out)
            time_5 = time.time()
            t_3 = time_5 - time_4




            # Individualization - Change corresponding label
            p = softmax(relu_out / 0.5, batch)
            combined = p.unsqueeze(1) * relu_out.unsqueeze(1)
            y = combined + y
            time_6 = time.time()
            t_4 = time_6 - time_5
            t_total = time_6 - time_2
            # print(t_1, t_2, t_3, t_4)
            # print(t_1/t_total, t_2/t_total, t_3/t_total, t_4/t_total)
            # sys.exit()
        # time_2 = time.time()
        # print(time_2 - time_1)
        # sys.exit()
        
        
        # Last refinement
        collect_info = self.InfoCollect_1(y, data.edge_index)
        collect_info = self.InfoCollect_2(collect_info, data.edge_index)
        collect_info = self.InfoCollect_3(collect_info, data.edge_index).t()

        X = collect_info.view(-1, 1)
        D = X - X.T
        D_1 = CustomSignFunction.apply(D, self.k, self.epsilon)
        D = D_1.sum(dim=1, keepdim=True)
        # print(D)
        y = self.f5(D, num_nodes_per_graph)

        # print(y)
        # sys.exit()

        # Last GIN layer
        res = self.gin(y, data.edge_index)
        res = global_add_pool(res, batch)

        return res, node_indices




# class SiameseNetwork(nn.Module):
#     def __init__(self, ISGNN, device):
#         super(SiameseNetwork, self).__init__()
#         self.custom_network = ISGNN
#         self.device = device

#     def forward(self, input1, input2):

#         time_0 = time.time()
#         output1, indices1 = self.custom_network(input1, self.device)
#         time_1 = time.time()
#         output2, indices2 = self.custom_network(input2, self.device)
#         # output = F.cosine_similarity(output1, output2, dim=1)

#         # squared_differences = (output1 - output2) ** 2  # Shape: (3, 5)

#         output = torch.norm(output1 - output2, p=2, dim=1)
#         # output = torch.sigmoid(output)  
#         # print(output)
#         # sys.exit()
        
#         indices1 = torch.stack(indices1)
#         indices2 = torch.stack(indices2)
#         # print(f'twin model forward time: {time.time()-time_0}')s
        
#         return output, output1, output2, indices1, indices2
    
#     def reset_parameters(self):
#         for name, param in self.named_parameters():
#             if param.dim() >= 2:
#                 nn.init.xavier_uniform_(param)
#             elif param.dim() == 1:
#                 nn.init.uniform_(param, a=-10, b=10)
#             elif param.dim() == 0:
#                 print(f"Warning: {name} is a scalar parameter with value {param.item()}. Skipping initialization.")
#             else:
#                 raise ValueError(f"Unexpected parameter '{name}' with dimension {param.dim()} and shape {param.shape}")





class approxGNN(nn.Module):
    def __init__(self, device, hyperpara1, hyperpara2, local_in, local_out, dname):
        super(approxGNN, self).__init__()
        self.hyper1 = hyperpara1
        self.hyper2 = hyperpara2
        self.device = device
        self.dname = dname
        self.k = 1000
        self.epsilon = 5
        self.local = InfoCollect(local_in, local_out)


    def find_nodes_with_most_common_label(self, partition):
        """
        Finds the indexes of nodes whose partition label is the most common.

        Args:
            partition (list or array-like): A list of partition labels for each node.

        Returns:
            list: A list of node indexes with the most common partition label.
        """
 

        # Step 1: Count the occurrences of each partition label
        label_counts = Counter(partition)

        # Step 2: Identify the most common partition label
        most_common_label, max_count = label_counts.most_common(1)[0]

        # Step 3: Find the node indexes with the most common label
        node_indexes = [index for index, label in enumerate(partition) if label == most_common_label]

        return node_indexes

    def generate_one_hot_vectors(self, indices, n):
        """
        Generates a list of one-hot tensors, one for each index in 'indices'.

        Args:
            indices (list or array-like): List of indices for which to generate one-hot vectors.
            n (int): Size of each one-hot vector (number of nodes).

        Returns:
            list of torch.Tensor: A list where each element is a one-hot tensor of size [n] with a single 1 at index i.
        """
        one_hot_vectors = []
        for i in indices:
            a = torch.zeros(n)
            a[i] = 1
            one_hot_vectors.append(a)
        return one_hot_vectors



    def refine(self, info, data):
        # print(device)
        fixed = False
        edge_index = data.edge_index
        num_nodes = data.x.shape[0]
        time = 0
        while not fixed and time <= self.hyper1:
            X = self.local(info, edge_index).view(-1, 1)
            print(X)
            print(X.shape)
            D = X - X.T
            D_1 = CustomSignFunction.apply(D, self.k, self.epsilon)
            D = D_1.sum(dim=1, keepdim=True)
            new_p = self.f5(D, num_nodes).squeeze()

            if torch.equal(new_p, prev):
                fixed = True
            else:
                prev = new_p
            time += 1
        new_p = new_p.squeeze()
        return new_p




    def forward(self, data):
        num_nodes = data.x.shape[0]
        p_0 = self.refine(data.x, data)

        indi_one_hot = self.generate_one_hot_vectors(self.find_nodes_with_most_common_label(p_0), num_nodes)

        candidate = []
        for i in range(num_nodes):
            for indi_vector in indi_one_hot[:int(self.hyper2*len(indi_one_hot))]:
                p_tempt = p_0 + indi_vector
                p_current = self.refine(p_tempt, data)
                candidate.append(p_current)









        
        











class SiameseNetwork(nn.Module):
    def __init__(self, device, prop, dname):
        super(SiameseNetwork, self).__init__()
        self.custom_network = ExactModel(prop=prop)
        self.device = device
        self.dname = dname
        # print(self.device)
    def forward(self, input1, input2, labels):

        time_0 = time.time()
        # print(self.device)
        # print(labels)
        # sys.exit()
        label = int(labels.item())
        # print(label)
        # sys.exit()
        output1, leaves1 = self.custom_network(input1, self.device)
        time_1 = time.time()
        output2, leaves2 = self.custom_network(input2, self.device)
        G_1 = to_networkx(input1, to_undirected=True)
        G_2 = to_networkx(input2, to_undirected=True)
        # print(output1)
        # print(output2)
        a = check_iso(output1.tolist(), output2.tolist(), G_1, G_2, label)
        # print(a)
        if not a:
            
            # print('==================================================1')
            # print(G_1.edges())
            # # torch.save(input1.edge_index, 'edge_index_1.pt')
            leaves_converted_1 = {key: [tensor.tolist() for tensor in value] for key, value in leaves1.items()}
            leaves_converted_2 = {key: [tensor.tolist() for tensor in value] for key, value in leaves2.items()}

            for (partition_1, score_1), (partition_2, score_2) in zip(leaves_converted_1.items(), leaves_converted_2.items()):
                # print(type(score_1))

                if score_1 == score_2:
                    
                    print('equal')
                elif score_1 > score_2:
                    print('partition1 large')
                else:
                    print('partition2 large')

                # print(input1.edge_index)
                # print(score_1)
            # print('==================================================2')
            # file_name = '/home/cds/Documents/Yifan/GI_Project/find_why.txt'

            # with open(file_name, "w") as file:
            #     file.write('G_1 \n')
            #     for edge in G_1.edges():
            #         file.write(str(edge) + '\n')
            #     file.write('G_1 cl \n')
            #     file.write(str(output1.tolist()) + '\n')


                
            #     file.write('G_2 \n')
            #     for edge in G_2.edges():
            #         file.write(str(edge) + '\n')
            #     file.write('G_2 cl \n')
            #     file.write(str(output2.tolist()))


                


            # print(G_2.edges())
            # # torch.save(input2.edge_index, 'edge_index_2.pt')


            # for (partition_1, score_1), (partition_2, score_2) in zip(leaves1.items(), leaves2.items()):
            #     print(partition_2.tolist())
                # print(score_2)
            
            # G_1 = nx.Graph()
            # G_2 = nx.Graph()
            # nodes_list = [i for i in range(100)]
            # G_1.add_nodes_from(nodes_list)
            # G_2.add_nodes_from(nodes_list)

            # edges1 = input1.edge_index.t().tolist()  # Transpose and convert to list of edges
            # G_1.add_edges_from(edges1)

            # edges2 = input2.edge_index.t().tolist()  # Transpose and convert to list of edges
            # G_2.add_edges_from(edges2)

            # cl_lst = output1.tolist()
            # cl_2_lst = output2.tolist()


            # label_1 = {}
            # label_2 = {}
            # for i in range(len(output1)):
            #     label_1[i] = cl_lst[i]
            #     label_2[i] = cl_2_lst[i]


            # draw_two_graphs(G_1, G_2, label_1, label_2)
            # # Save the first graph in adjacency list format
            # nx.write_adjlist(G_1, "graph1.adjlist")

            # # Save the second graph in adjacency list format
            # nx.write_adjlist(G_2, "graph2.adjlist")
            # sys.exit()
            return False
        # output = F.cosine_similarity(output1, output2, dim=1)
        # squared_differences = (output1 - output2) ** 2  # Shape: (3, 5)
        # output = torch.norm(output1 - output2, p=2, dim=-1)
        # output = torch.sigmoid(output)  
        # print(output)
        # sys.exit()
        # indices1 = torch.stack(indices1)
        # indices2 = torch.stack(indices2)
        # print(f'twin model forward time: {time.time()-time_0}')s
        return True
    
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                nn.init.uniform_(param, a=-10, b=10)
            elif param.dim() == 0:
                print(f"Warning: {name} is a scalar parameter with value {param.item()}. Skipping initialization.")
            else:
                raise ValueError(f"Unexpected parameter '{name}' with dimension {param.dim()} and shape {param.shape}")




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MyCustomData(Data):
    def __init__(self, node_features, edge_index, idx=None, **kwargs):
        super().__init__(x=node_features, edge_index=edge_index, **kwargs)
        self.idx = idx
        self.num_nodes = node_features.size(0)
    # You can also define custom methods to operate on your data
    def custom_method(self):
        # Example method that does something with custom_attr
        return self.custom_attr.mean() if self.custom_attr is not None else None

    def __inc__(self, key, value, *args, **kwargs):
        # Handle how to increment custom attributes when batching
        if key == 'idx':
            return 1  # Adjust increment strategy if batching
        else:
            return super().__inc__(key, value, *args, **kwargs)