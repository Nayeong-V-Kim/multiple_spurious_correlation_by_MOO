import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from data.confounder_dataset import ConfounderDataset


class MultiCelebADataset(ConfounderDataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(self, root_dir, target_name, confounder_names,
                 model_type):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type

        # Read in attributes
        self.attrs_df = pd.read_csv(
            os.path.join(root_dir, 'list_attr_celeba.csv'))
        self.data_dir = os.path.join(self.root_dir, 'img_align_celeba')
        self.filename_array = self.attrs_df['image_id'].values
        self.attrs_df = self.attrs_df.drop(labels='image_id', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()
        
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0
        
        # Get the y values
        # target_idx = self.attr_idx(self.target_name)
        target_idx = self.attr_idx('High_Cheekbones')
        
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2
        
        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        
        
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        self.confounder_array = confounder_id

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')
        
        self.split_df = pd.read_csv(
            os.path.join(root_dir, 'list_eval_partition.csv'))
        self.split_array = self.split_df['partition'].values
        #######
        train_arr = self.attrs_df[:162770]
        young = train_arr[:, self.attr_idx('Young')]
        male = train_arr[:, self.attr_idx('Male')]
        cheekbones = train_arr[:, self.attr_idx('High_Cheekbones')]
        train_idx = []
    
        train_idx.extend(np.where((young==1)*(male==0)*(cheekbones==1))[0]) # 44528
        train_idx.extend(np.where((young==1)*(male==1)*(cheekbones==1))[0][:2200])
        train_idx.extend(np.where((young==0)*(male==0)*(cheekbones==1))[0][:2200])
        train_idx.extend(np.where((young==0)*(male==1)*(cheekbones==1))[0][:110])

        train_idx.extend(np.where((young==1)*(male==0)*(cheekbones==0))[0][:40])
        train_idx.extend(np.where((young==1)*(male==1)*(cheekbones==0))[0][:800])
        train_idx.extend(np.where((young==0)*(male==0)*(cheekbones==0))[0][:800])
        train_idx.extend(np.where((young==0)*(male==1)*(cheekbones==0))[0][:16220])

        self.split_array[train_idx] = 3

        #######
        
        self.split_dict = {
            'train': 3,
            'val': 1,
            'test': 2,
            'etc': 4
        }
        # from collections import Counter
        # set([*range(64)])-set(sorted(Counter(loader.dataset.dataset.dataset.group_array[loader.dataset.dataset.dataset.split_array==2])))
        if model_attributes[self.model_type]['feature_type']=='precomputed':
            self.features_mat = torch.from_numpy(np.load(
                os.path.join(root_dir, 'features', model_attributes[self.model_type]['feature_filename']))).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_celebA(self.model_type, train=True)
            self.eval_transform = get_transform_celebA(self.model_type, train=False)

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)


def get_transform_celebA(model_type, train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    if model_attributes[model_type]['target_resolution'] is not None:
        target_resolution = model_attributes[model_type]['target_resolution']
    else:
        target_resolution = (orig_w, orig_h)

    transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transform
