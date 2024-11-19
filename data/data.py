import os
import numpy as np
from data.confounder_utils import prepare_confounder_data

root_dir = ''

dataset_attributes = {
    'MultiCelebA': {
        'root_dir': '/{your_root_dir}/celeba'
    },
    'UrbanCars': {
        'root_dir' : "/{your_root_dir}/Whac-A-Mole-main/data"
    },
    'MultiMNIST': {
        'root_dir': '/{your_root_dir}'
    },
    'CelebA': {
        'root_dir': '/{your_root_dir}/celeba'
    },
    'Waterbirds': {
        'root_dir' : "/{your_root_dir}/waterbirds/waterbird_complete95_forest2water2"
    },
    'bFFHQ': {
        'root_dir' : "/{your_root_dir}/bffhq/0.5pct"
    }
}
for dataset in dataset_attributes:
    dataset_attributes[dataset]['root_dir'] = os.path.join(root_dir, dataset_attributes[dataset]['root_dir'])

shift_types = ['confounder', 'label_shift_step']

def prepare_data(args, train, return_full_dataset=False):
    # Set root_dir to defaults if necessary
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']
    if args.shift_type=='confounder':
        return prepare_confounder_data(args, train, return_full_dataset)
    
def log_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        # logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
        logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
        # logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            # logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')
            logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')