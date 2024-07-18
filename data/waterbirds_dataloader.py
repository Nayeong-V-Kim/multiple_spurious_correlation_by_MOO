import os
import numpy as np
from torchvision import transforms
import torch
from torch.utils import data
from PIL import Image
import pandas as pd
from data.confounder_dataset import ConfounderDataset
from models import model_attributes

class Waterbirds(ConfounderDataset):
    def __init__(self, root_dir, target_name, confounder_names, train='train', select=False, model_type=None):
        self.data_dir = root_dir
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))
        self.file_list = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.confounder_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.y_array = self.metadata_df['place'].values

        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.n_confounders = self.n_groups
        self.filename_array = self.metadata_df['img_filename'].values
        scale = 256.0/224.0
        target_resolution = model_attributes[model_type]['target_resolution'] # 224
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type

        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.train_transform =  transforms.Compose([
                            transforms.RandomResizedCrop(
                                target_resolution, #[224,224],
                                scale=(0.7, 1.0),
                                ratio=(0.75, 1.3333333333333333),
                                interpolation=2),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        self.eval_transform = transforms.Compose([
                        transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))), #transforms.Resize([256,256]),
                        transforms.CenterCrop(target_resolution),# transforms.CenterCrop([224,224]),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    def __len__(self):
        return len(self.data)
    