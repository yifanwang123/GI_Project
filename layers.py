import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGEConv, GINConv, GATConv
from torch_geometric.utils import to_dense_adj
import torch_geometric.nn as pyg_nn

from torch_geometric.utils import softmax
from torch.nn import LeakyReLU


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  # Batch normalization after fc1
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)  # Batch normalization after fc2
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply BatchNorm after each linear layer
        x = self.relu(self.bn1(self.fc1(x)))  # Batch normalization after fc1 and ReLU
        x = self.bn2(self.fc2(x))  # Batch normalization after fc2
        return x

# GINModel that supports dynamic number of layers and hidden dimensions
class GINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels, num_layers):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.gin_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()  # BatchNorm layers for each GINConv

        # First GINConv layer
        mlp = MLP(in_channels, hidden_channels_list, hidden_channels_list)
        self.gin_layers.append(pyg_nn.GINConv(mlp))
        self.bn_layers.append(nn.BatchNorm1d(hidden_channels_list))  # BatchNorm after first GINConv

        # Intermediate GINConv layers
        for i in range(1, num_layers - 1):
            mlp = MLP(hidden_channels_list, hidden_channels_list, hidden_channels_list)
            self.gin_layers.append(pyg_nn.GINConv(mlp))
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels_list))  # BatchNorm after each intermediate GINConv

        # Final GINConv layer
        mlp = MLP(hidden_channels_list, hidden_channels_list, hidden_channels_list)
        self.gin_layers.append(pyg_nn.GINConv(mlp))
        self.bn_layers.append(nn.BatchNorm1d(hidden_channels_list))  # BatchNorm after the final GINConv

        # Final linear layer for output
        self.fc = nn.Linear(hidden_channels_list, out_channels)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.gin_layers[i](x, edge_index)
            x = self.bn_layers[i](x)  # Apply BatchNorm after each GINConv layer
            x = F.relu(x)
        x = self.fc(x)  # Final linear transformation
        return x




class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=8, num_heads=8, dropout=0.6):
        super(GATNet, self).__init__()
        self.dropout = dropout

        # First GAT layer (multi-head)
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate heads' outputs
        )

        # Second GAT layer (single head)
        self.gat2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            concat=False  # Do not concatenate since heads=1
        )

    def forward(self, x, edge_index):
        # Apply dropout to input features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # First GAT layer with activation function
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GAT layer
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)






class FixedWeightMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(FixedWeightMessagePassing, self).__init__(aggr='add')  # 'add' aggregation
        # Initialize a fixed weight matrix
        self.fixed_weight = torch.nn.Parameter(
            torch.ones(out_channels, in_channels), requires_grad=False
        )

    def forward(self, x, edge_index):
        # x: Node feature matrix [num_nodes, in_channels]
        # edge_index: Graph connectivity [2, num_edges]
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j: Feature matrix of neighboring nodes
        # Apply the fixed weight transformation
        return F.linear(x_j, self.fixed_weight)


class InfoCollect(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(InfoCollect, self).__init__(aggr='add')  # Sum aggregator
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(1, 1))
        self.attention = nn.Parameter(torch.Tensor(1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index):
        # print(x.shape)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i, edge_index):
        # x_j: Features of neighbor nodes
        # x_i: Features of target nodes

        # Compute attention score based only on the target node (x_i)

        # print(x_j.shape)
        # print(x_i.shape)
        # self.attention = self.attention.to(x_i.dtype)
        # print(self.attention.dtype)
        # print(x_i.dtype)
        x_i = x_i.to(self.attention.dtype)
        # print(self.attention.dtype)
        # print(x_i.dtype)
        alpha = torch.matmul(x_i, self.attention).squeeze(-1)
        alpha = torch.softmax(alpha, dim=0)  # Normalize attention scores

        # Apply shared attention score and weights to neighbor features
        x_j = x_j.to(self.attention.dtype)

        return alpha.unsqueeze(-1) * torch.matmul(x_j, self.weight)

    def update(self, aggr_out):
        return aggr_out  # Identity activation function


class IndiiLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(IndiiLayer, self).__init__(aggr='add')  # Sum aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Shared weight matrix for all nodes (U^l)
        self.U = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Attention mechanism based on the target node
        self.attention = nn.Parameter(torch.Tensor(in_channels, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i, edge_index):

        alpha = torch.matmul(x_i, self.attention).squeeze(-1)
        alpha = torch.softmax(alpha, dim=0)  # Normalize attention scores

        # Apply shared attention score and weights to neighbor features
        return alpha.unsqueeze(-1) * torch.matmul(x_j, self.U)

    def update(self, aggr_out):
        return aggr_out  # Identity activation function

    # def forward_with_argmax(self, x, edge_index):
    #     # print(f'here1 {x.shape}')
    #     # Perform message passing and return both aggregated output and argmax_Ux_j
    #     aggr_out = self.forward(x, edge_index)
    #     # print('here2: ', aggr_out.shape)
    #     return aggr_out, self.argmax_out  # Returning both values


class TrainableHardThreshold(nn.Module):
    def __init__(self, scale=10.0):
        super(TrainableHardThreshold, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))  # Learnable scaling factor

    def forward(self, x):
        # Sigmoid-based approximation of the threshold function
        return torch.tanh(self.scale * x)

# class DynamicTrainableStepFunctionF5(nn.Module):
#     def __init__(self, scale=10.0):
#         super(DynamicTrainableStepFunctionF5, self).__init__()

#         # Initialize a learnable scale parameter for controlling the sharpness of the step
#         self.scale = nn.Parameter(torch.tensor(scale))  # Can be trainable or fixed

#     def forward(self, x, N):
#         """
#         x: Input tensor
#         N: Number of steps (derived from data)
#         """
#         # Define dynamic thresholds based on N
#         thresholds = [N + 1 - 2 * (i + 1) for i in range(N - 1)]
#         thresholds = torch.tensor(thresholds, dtype=torch.float32, device=x.device)

#         # Initialize output with ones (f_5(x) = 1 when x < t_0)
#         output = torch.ones_like(x)

#         # Apply sigmoid-based soft thresholds dynamically
#         for i in range(1, N):
#             # Sigmoid approximation for each step
#             output += (N - i) * torch.sigmoid(self.scale * (x - thresholds[i - 1]))

#         return output
    

# class DynamicTrainableStepFunctionF5(nn.Module):
#     def __init__(self, scale=10.0):
#         super(DynamicTrainableStepFunctionF5, self).__init__()
#         self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
#         self.thresholds_cache = {}
    
#     def forward(self, x, N):
#         device = x.device
#         dtype = x.dtype

#         # Define the range of your input values
#         x_min, x_max = x.min().item(), x.max().item()
#         x_min = N+1-2*N
#         x_max = N+1-2*1
#         # print(x_min, x_max, N)
#         # sys.exit()

#         # Subtract a small epsilon to adjust the maximum threshold
#         epsilon = 1e-6
#         # x_max_adjusted = x_max - epsilon
#         x_max_adjusted = x_max


#         # Check if thresholds are cached
#         if N not in self.thresholds_cache:
#             # Compute thresholds excluding the maximum input value
#             thresholds = torch.linspace(x_min, x_max_adjusted, N - 1, dtype=dtype, device=device)
#             self.thresholds_cache[N] = thresholds
#         else:
#             thresholds = self.thresholds_cache[N]

#         # Reshape thresholds for broadcasting
#         thresholds_shape = [1] * x.dim() + [N - 1]
#         thresholds = thresholds.view(thresholds_shape)

#         # Expand x to shape compatible for broadcasting
#         x_expanded = x.unsqueeze(-1)

#         # Compute differences and sigmoid values
#         differences = x_expanded - thresholds
#         sigmoid_values = torch.sigmoid(self.scale * differences)

#         # Sum over the last dimension
#         weighted_sums = torch.sum(sigmoid_values, dim=-1)

#         # Compute the final output
#         output = torch.floor(torch.ones_like(x) + weighted_sums + 1e-6)

#         return output





class DynamicTrainableStepFunctionF5(nn.Module):
    def __init__(self, alpha=10.0):
        super(DynamicTrainableStepFunctionF5, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.thresholds_cache = {}

    def forward(self, x, N):
        device = x.device
        dtype = x.dtype

        # Compute thresholds t_i = N + 1 - 2(i + 1)
        if N not in self.thresholds_cache:
            i_values = torch.arange(1, N, dtype=dtype, device=device)  # i from 1 to N-1
            thresholds = N + 1 - 2 * (i_values + 1)
            self.thresholds_cache[N] = thresholds
        else:
            thresholds = self.thresholds_cache[N]
        
        # Reshape thresholds for broadcasting
        thresholds_shape = [1] * x.dim() + [len(thresholds)]
        thresholds = thresholds.view(thresholds_shape)

        # Expand x to shape compatible for broadcasting
        x_expanded = x.unsqueeze(-1)

        # Compute differences and sigmoid values
        differences = x_expanded - thresholds  # x - t_i
        differences = differences.half()
        sigmoid_values = torch.sigmoid(self.alpha * differences)

        # Sum over the last dimension
        sum_sigmoids = torch.sum(sigmoid_values, dim=-1)  # Sum over i

        # Compute the final output
        output = 1 + sum_sigmoids  # Outputs between 1 and N

        # Round to the nearest integer (optional, for integer outputs)
        output = torch.round(output)

        return output








class DynamicTrainableStepFunctionF5_batch(nn.Module):
    def __init__(self, scale=1.0):
        super(DynamicTrainableStepFunctionF5_batch, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.thresholds_cache = {}
        self.coefficients_cache = {}

    def _get_thresholds_and_coefficients(self, N, dtype, device):
        """Helper function to get thresholds and coefficients with caching."""
        if N not in self.thresholds_cache:
            # Compute thresholds and coefficients for this specific N
            thresholds = N + 1 - 2 * torch.arange(1, N, dtype=dtype, device=device)
            coefficients = torch.arange(N - 1, 0, -1, dtype=dtype, device=device)
            self.thresholds_cache[N] = thresholds
            self.coefficients_cache[N] = coefficients
        return self.thresholds_cache[N], self.coefficients_cache[N]

    def forward(self, x, a):
        # a = [N]
        device = x.device
        dtype = x.dtype
        output = torch.empty_like(x)

        # Start at the first node
        start_idx = 0

        for N in a:
            # Select the appropriate segment of x
            end_idx = start_idx + N
            x_segment = x[start_idx:end_idx]

            # Fetch cached thresholds and coefficients
            thresholds, coefficients = self._get_thresholds_and_coefficients(N, dtype, device)

            # Reshape for broadcasting (this is done once per N, not per element in x_segment)
            thresholds_shape = [1] * x_segment.dim() + [N - 1]  # Shape: (1, ..., 1, N - 1)
            thresholds = thresholds.view(thresholds_shape)
            coefficients = coefficients.view(thresholds_shape)

            # Expand x_segment to be compatible for broadcasting and compute sigmoid values
            x_expanded = x_segment.unsqueeze(-1)  # Shape: x_segment.shape + (1,)
            differences = x_expanded - thresholds  # Shape: x_segment.shape + (N - 1,)
            sigmoid_values = torch.sigmoid(self.scale * differences)  # Same shape as differences

            # Compute the weighted sum over the last dimension
            weighted_sums = torch.sum(coefficients * sigmoid_values, dim=-1)  # Shape: x_segment.shape

            # Compute the final output for this segment
            output[start_idx:end_idx] = 1 + weighted_sums

            # Update the start index for the next segment
            start_idx = end_idx

        return output
    


class LocalInfo(MessagePassing):
    def __init__(self, in_channels):
        super(LocalInfo, self).__init__(aggr='add')  # Use "add" aggregation
        # Initialize the learnable weights w_v for each node

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()  # Ensure w_v is between 0 and 1
        )

    def forward(self, x, edge_index):
        # Aggregate messages
        out = self.propagate(edge_index, x=x)
        # Add self-loop messages
        out = x + out
        # Compute w_v for each node
        w_v = self.mlp(x)
        # Multiply by the learnable weight w_v
        out = w_v * out
        return out

    def message(self, x_j):
        return x_j
    


class ValueFrequencyAttention(nn.Module):
    def __init__(self):
        super(ValueFrequencyAttention, self).__init__()
    
    def forward(self, node_values):
        """
        node_values: Tensor of shape [num_nodes], representing the value of each node
        """
        num_nodes = node_values.size(0)
        
        # Ensure node_values are integers if they are categorical labels
        # If node_values are not integers, map them to integers
        
        # Step 1: Compute value counts using torch.bincount
        unique_values, inverse_indices = torch.unique(node_values, return_inverse=True)
        value_counts = torch.bincount(inverse_indices)
        
        # Step 2: Map counts back to nodes
        frequencies = value_counts[inverse_indices]
        
        # Step 3: Normalize frequencies to (0, 1)
        frequencies = frequencies.float()
        frequencies = frequencies / frequencies.max()
        frequencies = frequencies.clamp(min=0.0, max=1.0)
        
        return frequencies

