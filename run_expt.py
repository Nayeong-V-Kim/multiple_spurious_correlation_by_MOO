import os, csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from simple_classifier import *

from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, log_args
from train import train

import wandb

def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    
    parser.add_argument('--early_stop_v', type=int, default=50)

    parser.add_argument('--root_dir', default=None)

    parser.add_argument('--reweight_groups', action='store_true', default=False)
    
    parser.add_argument('--grouping_type', type=str, default='gc_group')
    
    # Model
    parser.add_argument(
        '--model',
        choices=model_attributes.keys(),
        default='resnet50')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha_lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha_step', type=int, default=10)
    
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--optimizer', type=str, default='SGD')
    # Misc
    parser.add_argument('--seed', type=int, default=64)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    
    args = parser.parse_args()
    if '3types' in args.target_name:
        args.confounder_names = ['Young', 'Male', 'Mouth_Slightly_Open']
    else:
        args.confounder_names = ['Young', 'Male']    
    check_args(args)

    if args.reweight_groups:
        rg = '_reweight_groups'
    else:
        rg = ''

    log_name = f'{args.lr}_{args.weight_decay}_{args.optimizer}_CLR_{args.alpha_lr}_{args.grouping_type}_m{args.alpha_step}_BS{args.batch_size}{rg}_{args.seed}'
    args.log_dir = args.log_dir + '/' + log_name

    if args.wandb:
        project_name = '{}'.format(args.dataset)

        wandb.init(
            config=args,
            project=project_name,
            anonymous='allow',
            name=log_name
        )

    mode='w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    log_args(args, logger)

    set_seed(args.seed)
    
    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True)

    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':12, 'pin_memory':True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    
    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)
    
    ## Initialize model
    pretrained = not args.train_from_scratch
    if args.dataset == 'bFFHQ':
        pretrained = False
    
    if model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet18':
        print("ResNet18 , pretrained: ", pretrained)
        model = torchvision.models.resnet18(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'MLP':
        model = get_simple_classifier('mlp', n_classes, False)
    else:
        raise ValueError('Model not recognized.')

    logger.flush()
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    epoch_offset=0
    
    train(model, criterion, data, logger, args, epoch_offset=epoch_offset)

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.target_name
    
if __name__=='__main__':
    main()
