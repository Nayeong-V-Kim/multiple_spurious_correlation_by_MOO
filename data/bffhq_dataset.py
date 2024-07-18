import os
import torch
import numpy as np
import torchvision.transforms as transforms
from data.confounder_dataset import ConfounderDataset
from glob import glob

class bFFHQDataset(ConfounderDataset):
    def __init__(self,root_dir,
                target_name, confounder_names,
                 model_type=None):
        self.root_dir = root_dir
        self.image2pseudo = {}
        self.target_name = 'age'
        self.confounder_names = 'gender'
        self.n_confounders = 1
        self.model_type = model_type
        self.data_dir = root_dir
        y_array = []
        group_array = []
        split_array = []
        filename_array = []
        for split in ['train', 'valid', 'test']:
            if split=='train':
                self.align = glob(os.path.join(root_dir, 'align',"*","*"))
                self.conflict = glob(os.path.join(root_dir, 'conflict',"*","*"))
                self.data = self.align + self.conflict
                split_n = 0
            elif split=='valid':
                self.data = glob(os.path.join(root_dir,'../valid',"*"))  
                split_n = 1         
            elif split=='test':
                self.data = glob(os.path.join(root_dir, '../test',"*"))
                split_n = 2
                
            for ele in self.data:
                attr = torch.LongTensor([int(ele.split('_')[-2]), int(ele.split('_')[-1].split('.')[0])])
                y_array.append(attr[0])
                group_array.append(attr[1]+attr[0]*2)
                split_array.append(split_n)
                filename_array.append(ele)
        
        self.y_array = np.array(y_array)
        self.group_array = np.array(group_array)
        self.n_groups = 4
        self.split_array = np.array(split_array)
        self.filename_array = filename_array    
        
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.n_classes = 2
        
        self.train_transform = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.RandomCrop(224, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ])
        self.eval_transform = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ])

    def __len__(self):
        return len(self.filename_array)

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        attr_name = self.confounder_names[0]
        group_name = f'{self.target_name} = {int(y)}, {attr_name} = {int(c)}'
        
        return group_name