import torch

from torch_geometric.nn.pool import global_max_pool
from torch_geometric.nn import radius_graph, GCNConv, PointGNNConv
from torch_geometric.data import DataLoader

from graph.build_graph import LivoxTinyDataset


class PointGNN(torch.nn.Module):
    def __init__(self, r_downsample, r_process):
        super().__init__()
        self.downsample_gnn = torch.nn.Sequential(
            GCNConv(3, 64),
            GCNConv(64, 64),
            global_max_pool
        )
        self.
        self.r_downsample = r_downsample
        self.r_process = r_process


    def forward(self, data):
        # Downsample the data first
        data_downsampled = self.downsample(data)

        # Run MPNN for a point-cloud with the downsampled data

        
    
    def downsample(self, data):
        data = radius_graph(data.pos, r=self.r_downsample, batch=data.batch, loop=True, max_num_neighbors=128)
        return self.downsample_gnn(data)

    def mpnn(self, data):
        