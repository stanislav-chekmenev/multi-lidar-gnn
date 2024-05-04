# MultiLidarGNN
An equivariant variational GNN approach for a multi-LiDAR dataset

### Installation

Firstly, run the standard installation command:

```bash
pip install -r requirements.txt
```

Next, you should install pytorch geometric's additional packages. The example below installs a CUDA 12.1 version of the package:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### Dataset

The Livox Simu-dataset can be downloaded [here](https://livox-wiki-en.readthedocs.io/en/latest/data_summary/dataset.html).
You should unpack the `anno` and `points` directories into `data/raw` directory, but use just around 50-100 frames, the rest is
too big for quick experimenting and will be added later.

