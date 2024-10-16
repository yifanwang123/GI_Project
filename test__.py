from torch_scatter import scatter_max
import torch

# Example isolated test for scatter_max
contributions_log = torch.rand(100, 10, device='cuda')
target_nodes = torch.randint(0, 50, (100,), device='cuda')
total_nodes = 50
print('here')
max_per_target, _ = scatter_max(contributions_log, target_nodes, dim=0, dim_size=total_nodes)
print("Scatter max output:", max_per_target)