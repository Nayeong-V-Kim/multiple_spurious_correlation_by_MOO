from .multimnist_dataset import MultiColorMNIST
from data.celebA_dataset import CelebADataset
from data.dro_dataset import DRODataset
from data.multimnist_dataset import MultiColorMNIST
from data.multi_celebA_dataset import MultiCelebADataset
from data.waterbirds_dataloader import Waterbirds
from data.urbancars_dataset import UrbanCars
from data.bffhq_dataset import bFFHQDataset

################
### SETTINGS ###
################

confounder_settings = {
    'MultiCelebA':{
        'constructor': MultiCelebADataset
    },
    'UrbanCars': {
        'constructor': UrbanCars
    },
    'MultiMNIST': {
        'constructor': MultiColorMNIST
    },
    'CelebA':{
        'constructor': CelebADataset
    },
    'Waterbirds': {
        'constructor': Waterbirds
    },
    'bFFHQ': {
        'constructor': bFFHQDataset
    }
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]['constructor'](
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model)
    
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets
