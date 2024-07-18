import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import collections
from data.confounder_dataset import ConfounderDataset

class MultiColorMNIST(ConfounderDataset):
    attribute_names = ["digit", "LColor", 'p']
    basename = 'multi_color_mnist'
    target_attr_index = 0
    left_color_bias_attr_index = 1
    right_color_bias_attr_index = 2
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 model_type=None):
        # super().__init__()
        left_color_skew = 0.01 # 0.05 #
        right_color_skew = 0.05 #0.05
        severity = 4
        split = 'train'
        transform = ToTensor()
        assert split in ['train', 'valid']
        assert left_color_skew in [0.005, 0.01, 0.02, 0.05]
        assert right_color_skew in [0.005, 0.01, 0.02, 0.05]
        assert severity in [1, 2, 3, 4]
        self.target_name = target_name
        self.confounder_names = confounder_names
        
        root_dir = os.path.join(root_dir, self.basename, f'ColoredMNIST-SkewedA{left_color_skew}-SkewedB{right_color_skew}-Severity{severity}')
        assert os.path.exists(root_dir), f'{root_dir} does not exist'

        data_path = os.path.join(root_dir, split, "images.npy")
        self.data = np.load(data_path)
        train_length = self.data.shape[0]
        data_path = os.path.join(root_dir, 'valid', "images.npy")
        self.data = np.concatenate([self.data, np.load(data_path)])
        test_length = self.data.shape[0] - train_length

        attr_path = os.path.join(root_dir, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))
        attr_path = os.path.join(root_dir, 'valid', "attrs.npy")
        self.attr = np.concatenate([self.attr, torch.LongTensor(np.load(attr_path))])

        self.split_array = np.array([0]*train_length+[2]*test_length)
        
        self.split_dict = {
            'train': 0,
            'val': 2,
            'test': 2
        }

        # timecomplexity
        self.n_groups = 40
        self.n_classes = 10
        
        
        confounder = np.zeros([self.attr.shape[0],2])
        confounder_array = np.zeros(self.attr.shape[0])
        
        ### Train multibias
        for i in range(10):
            idx = np.where(self.attr[:,0]==i)[0]
            c1_idx = self.attr[idx,1]==i
            c2_idx = self.attr[idx,2]==i
            confounder[idx][c1_idx, 0] = 1
            confounder[idx][c2_idx, 1] = 1
            
            confounder_array[idx[c1_idx*c2_idx]] = i*4
            confounder_array[idx[c1_idx*(~c2_idx)]] = i*4+1
            confounder_array[idx[(~c1_idx)*c2_idx]] = i*4+2
            confounder_array[idx[(~c1_idx)*(~c2_idx)]] = i*4+3
        
        self.target = self.attr[:,0]
        self.y_array = self.target
        self.bias = confounder
        self.confounder_idx = [self.bias[:,a] for a in [0,1]]
        self.n_confounders = len(self.confounder_idx)
        self.group_array = confounder_array
        self.transform = transform
        
    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, idx):
        image, target, bias = self.data[idx], self.target[idx], self.group_array[idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, target, bias