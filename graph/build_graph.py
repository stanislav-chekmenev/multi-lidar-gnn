import pandas as pd
import numpy as np
import torch


from pathlib import Path
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import NormalizeScale


class LivoxDataset(Dataset):

    def __init__(self, root, pre_transofrm=NormalizeScale(), size=32):
        super().__init__(root, pre_transform=pre_transofrm)
        self.size = size

    @property
    def raw_file_names(self):
        points_anno_path = Path()
        points = [points_anno_path.joinpath('points', '{:08}'.format(i) + '.txt') for i in range(self.size)]
        anno = [points_anno_path.joinpath('anno', '{:08}'.format(i) + '.txt') for i in range(self.size)]
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

            # Leave only the points from 6 Lidars
            points = points[(points['lidar_number'] == 1) | (points['lidar_number'] == 6)]

            # Remove unnecessary columns
            points = points[['x', 'y', 'z', 'type']]
            anno.drop(columns=['tracking_id'], inplace=True)


            # Create one x-variable for the graph holding all three coordinates
            pos = np.concatenate(
                [points['x'].values, points['y'].values, points['z'].values], axis=0
            ).reshape(3, -1).T

            # Create a y-variable for the graph holding all point types, which we need for the CE loss
            y = points['type'].values

            # Create a y-pos variable for the graph holding all annotation data for the bounding boxes, 
            # which we need for the Huber loss
            y_bb = np.concatenate(
                [
                    anno['x'].values, anno['y'].values, anno['z'].values,
                    anno['length'].values, anno['width'].values, anno['height'].values, anno['yaw'].values],
                axis=0
            ).reshape(7, -1).T

            # Create class labels for each bounding box
            y_bb_cls = anno['type'].values

            # Create a graph
            pos = torch.tensor(pos, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            y_bb = torch.tensor(y_bb, dtype=torch.float)
            data = Data(pos=pos, y=y, y_bb=y_bb, y_bb_cls=y_bb_cls)

            # Scale and normalize the data
            if self.pre_transform is not None:
                # TODO: find out how to properly normalize the data if it's even needed
                pass

            # Split into the train and test sets
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[:int(0.8 * data.num_nodes)] = 1
            data.test_mask = ~data.train_mask

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
    dataset = LivoxDataset('../data', size=32)
    print(dataset[0])
