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

### Datasets

- Let's use KITTI-360 and return to Livox only if it's neccessary. KITTI can be downloaded [here](https://www.cvlibs.net/datasets/kitti-360/), but
first you'd need to register on their website and explain how you'd like to use the dataset. Extract the data into `data/kitti360/raw`directory.

- The Livox Simu-dataset can be downloaded [here](https://livox-wiki-en.readthedocs.io/en/latest/data_summary/dataset.html).
You should unpack the `anno` and `points` directories into `data/livox/raw` directory.

