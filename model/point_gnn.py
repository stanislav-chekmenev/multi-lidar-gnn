import torch

from torch_geometric.nn.pool import global_max_pool
from torch_geometric.nn import radius_graph, MessagePassing
from torch_geometric.data import DataLoader

from graph.build_graph import LivoxTinyDataset


class PointNetPP(MessagePassing):
    def __init__(self, r_downsample, r_process):
        pass