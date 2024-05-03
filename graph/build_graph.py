import pandas as pd
import numpy as np
import torch


from pathlib import Path
from torch_geometric.data import Dataset, Data


class LivoxTinyDataset(Dataset):

    def __init__(self, root):
        super().__init__(root)

    @property
    def raw_file_names(self):
        points_anno_path = Path()
        points = [points_anno_path.joinpath('points', '{:08}'.format(i) + '.txt') for i in range(32)]
        anno = [points_anno_path.joinpath('anno', '{:08}'.format(i) + '.txt') for i in range(32)]
        return (points, anno)

    @property
    def processed_file_names(self):
        graphs = [f'graph_{i}.pt' for i in range(32)]
        return graphs
    
    def process(self):
        # Read each point and annotation file
        idx = 0

        for raw_points, raw_anno in zip(*self.raw_file_names):
            print(f'Processing {idx}th file')

            # Read the raw data
            raw_dir_path = Path(self.raw_dir)
            points = pd.read_csv(raw_dir_path.joinpath(raw_points), header=None, sep=',')
            anno = pd.read_csv(raw_dir_path.joinpath(raw_anno), header=None, sep=',')

            points_column_names = ['x', 'y', 'z', 'motion_state', 'type', 'lidar_number']
            anno_column_names =  ['tracking_id', 'type', 'x', 'y', 'z', 'length', 'width', 'height', 'yaw']

            points.columns = points_column_names
            anno.columns = anno_column_names

            # Remove unnecessary columns
            points = points[['x', 'y', 'z']]
            anno.drop(columns=['tracking_id', 'type'], inplace=True)

            # Create one x-variable for the graph holding all three coordinates
            pos = np.concatenate(
                [points['x'].values, points['y'].values, points['z'].values], axis=0
            ).reshape(3, -1).T

            # Create one y-variable for the graph holding all annotation data
            y = np.concatenate(
                [
                    anno['x'].values, anno['y'].values, anno['z'].values,
                    anno['length'].values, anno['width'].values, anno['height'].values, anno['yaw'].values],
                axis=0
            ).reshape(7, -1).T

            # Create a graph
            pos = torch.tensor(pos, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            data = Data(x=pos, y=y)

            # Save the processed data
            processed_dir_path = Path(self.processed_dir)
            torch.save(data, processed_dir_path.joinpath(f'graph_{idx}.pt'))

            # Proceed to the next file
            idx += 1
            
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        processed_dir_path = Path(self.processed_dir)
        data = torch.load(processed_dir_path.joinpath(f'graph_{idx}.pt'))
        return data
    
if __name__ == '__main__':
    dataset = LivoxTinyDataset('../data')
    print(dataset)
