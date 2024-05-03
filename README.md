# MultiLidarGNN
An equivariant variational GNN approach for a multi-LiDAR dataset


### Installation

Firstly, run the standard installation command:

```bash
pip install -r requirements.txt
```

Next, you should install pytorch geometric. The example below install a CUDA 12.1 version of the package:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```