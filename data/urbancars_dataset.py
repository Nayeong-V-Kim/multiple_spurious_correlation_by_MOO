
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import glob
import torch
import random
import numpy as np

from PIL import Image
from data.confounder_dataset import ConfounderDataset

import torchvision.transforms as transforms

class UrbanCars(ConfounderDataset):
    base_folder = "urbancars"
    
    obj_name_list = [
        "urban",
        "country",
    ]

    bg_name_list = [
        "urban",
        "country",
    ]

    co_occur_obj_name_list = [
        "urban",
        "country",
    ]

    def __init__(
        self,
        root_dir,
        target_name, confounder_names,
        group_label="both",
        model_type=None,
        return_group_index=False,
        return_domain_label=False,
        return_dist_shift=False
    ):
        assert group_label in ["bg", "co_occur_obj", "both"]
        self.train_transform = get_transforms('train', target_name)
        self.eval_transform = get_transforms('val', target_name)
        
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.data_dir = ''
        
        self.return_group_index = return_group_index
        self.return_domain_label = return_domain_label
        self.return_dist_shift = return_dist_shift

        assert os.path.exists(os.path.join(root_dir, self.base_folder))

        self.filename_array = []
        self.obj_bg_co_occur_obj_label_list = []
        split_dict = {'train': [0, 0.95, 0.95], 'val': [1, 0.5, 0.5], 'test': [2, 0.5, 0.5]}
        self.split_dict = {'train': 0,'val': 1,'test': 2}
        self.split_array = []
        
        for split in ['train', 'val', 'test']:
            split_idx, bg_ratio, co_occur_obj_ratio = split_dict[split]
            ratio_combination_folder_name = (
                f"bg-{bg_ratio}_co_occur_obj-{co_occur_obj_ratio}"
            )
            img_root = os.path.join(
                root_dir, self.base_folder, ratio_combination_folder_name, split
            )
            for obj_id, obj_name in enumerate(self.obj_name_list):
                for bg_id, bg_name in enumerate(self.bg_name_list):
                    for co_occur_obj_id, co_occur_obj_name in enumerate(
                        self.co_occur_obj_name_list
                    ):
                        dir_name = (
                            f"obj-{obj_name}_bg-{bg_name}_co_occur_obj-{co_occur_obj_name}"
                        )
                        dir_path = os.path.join(img_root, dir_name)
                        assert os.path.exists(dir_path)

                        filename_array = glob.glob(os.path.join(dir_path, "*.jpg"))
                        self.filename_array += filename_array

                        self.obj_bg_co_occur_obj_label_list += [
                            (obj_id, bg_id, co_occur_obj_id)
                        ] * len(filename_array)
                        self.split_array += [split_idx]*(len(filename_array))
        
        self.split_array = np.array(self.split_array)
        self.obj_bg_co_occur_obj_label_list = torch.tensor(
            self.obj_bg_co_occur_obj_label_list, dtype=torch.long
        )

        self.obj_label = self.obj_bg_co_occur_obj_label_list[:, 0]
        bg_label = self.obj_bg_co_occur_obj_label_list[:, 1]
        co_occur_obj_label = self.obj_bg_co_occur_obj_label_list[:, 2]

        if group_label == "bg":
            num_shortcut_category = 2
            shortcut_label = bg_label
        elif group_label == "co_occur_obj":
            num_shortcut_category = 2
            shortcut_label = co_occur_obj_label
        elif group_label == "both":
            num_shortcut_category = 4
            shortcut_label = bg_label * 2 + co_occur_obj_label
        else:
            raise NotImplementedError

        self.domain_label = shortcut_label
        
        self.set_num_group_and_group_array(num_shortcut_category, shortcut_label)
        
        self.y_array = self.obj_bg_co_occur_obj_label_list[:,0]
        self.n_classes = 2
        self.n_groups = 8
        
        self._group_array = self.group_array
        self.n_confounders = 2
        
    def _get_subsample_group_indices(self, subsample_which_shortcut):
        bg_ratio = self.bg_ratio
        co_occur_obj_ratio = self.co_occur_obj_ratio

        num_img_per_obj_class = len(self) // len(self.obj_name_list)
        if subsample_which_shortcut == "bg":
            min_size = int(min(1 - bg_ratio, bg_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "co_occur_obj":
            min_size = int(min(1 - co_occur_obj_ratio, co_occur_obj_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "both":
            min_bg_ratio = min(1 - bg_ratio, bg_ratio)
            min_co_occur_obj_ratio = min(1 - co_occur_obj_ratio, co_occur_obj_ratio)
            min_size = int(min_bg_ratio * min_co_occur_obj_ratio * num_img_per_obj_class)
        else:
            raise NotImplementedError

        assert min_size > 1

        indices = []

        if subsample_which_shortcut == "bg":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
                for idx_bg in range(len(self.bg_name_list)):
                    bg_mask = self.obj_bg_co_occur_obj_label_list[:, 1] == idx_bg
                    mask = obj_mask & bg_mask
                    subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                    random.shuffle(subgroup_indices)
                    sampled_subgroup_indices = subgroup_indices[:min_size]
                    indices += sampled_subgroup_indices
        elif subsample_which_shortcut == "co_occur_obj":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
                for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                    co_occur_obj_mask = self.obj_bg_co_occur_obj_label_list[:, 2] == idx_co_occur_obj
                    mask = obj_mask & co_occur_obj_mask
                    subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                    random.shuffle(subgroup_indices)
                    sampled_subgroup_indices = subgroup_indices[:min_size]
                    indices += sampled_subgroup_indices
        elif subsample_which_shortcut == "both":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
                for idx_bg in range(len(self.bg_name_list)):
                    bg_mask = self.obj_bg_co_occur_obj_label_list[:, 1] == idx_bg
                    for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                        co_occur_obj_mask = self.obj_bg_co_occur_obj_label_list[:, 2] == idx_co_occur_obj
                        mask = obj_mask & bg_mask & co_occur_obj_mask
                        subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                        random.shuffle(subgroup_indices)
                        sampled_subgroup_indices = subgroup_indices[:min_size]
                        indices += sampled_subgroup_indices
        else:
            raise NotImplementedError

        return indices

    def set_num_group_and_group_array(self, num_shortcut_category, shortcut_label):
        self.num_group = len(self.obj_name_list) * num_shortcut_category
        self.group_array = self.obj_label * num_shortcut_category + shortcut_label

    def set_domain_label(self, shortcut_label):
        self.domain_label = shortcut_label

    def __len__(self):
        return len(self.filename_array)

    def get_labels(self):
        return self.obj_bg_co_occur_obj_label_list

    def get_sampling_weights(self):
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array]
        return weights

def get_transforms(split, target_name):
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    if split=='train':
        aug_list = []
        target_resolution = 256
        if 'crop' in target_name:
            aug_list.append(transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=2))
        if 'color' in target_name:
            aug_list.append(transforms.ColorJitter(brightness=(0.5, 0.9), 
                            contrast=(0.4, 0.8), 
                            saturation=(0.7, 0.9),
                            hue=(-0.2, 0.2),
                            ))
        if 'rotate' in target_name:
            aug_list.append(transforms.RandomRotation(degrees=(-30, 30), interpolation=transforms.InterpolationMode.BILINEAR, fill=0))
        
        if 'jw' in target_name:
            if 'soft' in target_name:
                soft = True
            else:
                soft = False
            
            aug_list = []
            aug_list.append(transforms.Pad(padding=int(target_resolution / 10), padding_mode='edge'))
            aug_list.append(transforms.RandomAffine(
                    degrees=[-8, 8] if soft else [-15, 15],
                    translate=(1/16, 1/16),
                    scale=(0.95, 1.05) if soft else (0.9, 1.1),
                    shear=None,
                    resample=Image.BILINEAR,
                    fillcolor=None
                ))
            aug_list.append(transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]))
            aug_list.append(transforms.CenterCrop(size=target_resolution))
            
        aug_list.append(transforms.RandomHorizontalFlip())
        aug_list.append(transforms.ToTensor())
        aug_list.append(normalize)
        transform = transforms.Compose(aug_list)
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )


    return transform
