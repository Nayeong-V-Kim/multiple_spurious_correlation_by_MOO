import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer
import copy

import wandb

import time
# from sam import SAM
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, args,
              is_training, show_progress=False, log_every=50,
              group_alpha=None, loss_lambda=None,
              optimizer_alpha=None, optimizer_lambda=None, log_mode='train', 
              run_wandb=False, momentum_grad=None):

    if epoch ==0:
        GLOBAL_COUNTER = 0
    else:
        GLOBAL_COUNTER = 1
    if is_training:
        model.train()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader
    
    metrics = {}
    with torch.set_grad_enabled(is_training):
        group_list = []
        for batch_idx, batch in enumerate(prog_bar_loader):
            if is_training and batch_idx > 300 and args.dataset == 'CelebA':
                break
            # if is_training and batch_idx > 20 and args.dataset == 'MultiMNIST':
                # break
            # if is_training and batch_idx >= 10 and args.dataset == 'MultiCelebA':
            #     break
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            
            outputs = model(x)

            loss_main, metrics, momentum_grad = loss_computer.loss(outputs, y, g, is_training, optimizer=optimizer,
                                            group_alpha=group_alpha, loss_lambda=loss_lambda, metrics=metrics, momentum_grad=momentum_grad, alpha_step=(batch_idx % args.alpha_step ==0))
                
            if is_training:
                if batch_idx % args.alpha_step ==0:
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_alpha.zero_grad()
                    optimizer_lambda.zero_grad()
                    loss_main.backward()
                    optimizer_alpha.step()
                    loss_lambda.grad *= -1
                    optimizer_lambda.step()
                            
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                for i in range(group_alpha.shape[0]):
                    log_alpha = copy.deepcopy(group_alpha.data.cpu())
                    metrics['group_pure_alpha{}'.format(i)] = log_alpha[i]
                    log_alpha = F.softmax(log_alpha)
                    metrics['group_alpha{}'.format(i)] = log_alpha[i]
                metrics['loss_lambda'] = loss_lambda.cpu().data
                
            if is_training and ((batch_idx+1) % log_every==0) and (batch_idx+1 != prog_bar_loader.__len__()):
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

            if run_wandb:
                wandb.log(metrics)
        
        if (not is_training) or loss_computer.batch_count > 0:
            metrics = {}
            stats = loss_computer.get_stats(model, args)
            if args.dataset == 'MultiCelebA' or args.dataset == 'UrbanCars':
                if '3types' in args.target_name:
                    acc = min(stats['avg_acc_group:{}'.format(i)] for i in range(16))
                else:
                    acc = min(stats['avg_acc_group:{}'.format(i)] for i in range(8))
                    if args.dataset == 'MultiCelebA':
                        stats['gg'] = (stats['avg_acc_group:2'] + stats['avg_acc_group:5'])/2
                        stats['gc'] = (stats['avg_acc_group:3'] + stats['avg_acc_group:4'])/2
                        stats['cg'] = (stats['avg_acc_group:0'] + stats['avg_acc_group:7'])/2
                        stats['cc'] = (stats['avg_acc_group:1'] + stats['avg_acc_group:6'])/2
                # worst = min(stats['avg_acc_group:{}'.format(i)] for i in range(8))
            elif args.dataset == 'Waterbirds' or args.dataset == 'CelebA':
                acc = min(stats['avg_acc_group:{}'.format(i)] for i in range(4))
            elif args.dataset == 'MultiMNIST':
                index = np.array([*range(10)])*4
                gg=np.array([stats['avg_acc_group:{}'.format(a)] for a in index]).mean()
                gg_loss=np.array([stats['avg_loss_group:{}'.format(a)] for a in index]).mean()
                index = np.array([*range(10)])*4+1
                gc=np.array([stats['avg_acc_group:{}'.format(a)] for a in index]).mean()
                gc_loss=np.array([stats['avg_loss_group:{}'.format(a)] for a in index]).mean()
                index = np.array([*range(10)])*4+2
                cg=np.array([stats['avg_acc_group:{}'.format(a)] for a in index]).mean()
                cg_loss=np.array([stats['avg_loss_group:{}'.format(a)] for a in index]).mean()
                index = np.array([*range(10)])*4+3
                cc=np.array([stats['avg_acc_group:{}'.format(a)] for a in index]).mean()
                cc_loss=np.array([stats['avg_loss_group:{}'.format(a)] for a in index]).mean()
                # acc = min([gg, cg, gc, cc])
                acc = cc
                stats['gg'] = gg
                stats['gc'] = gc
                stats['cg'] = cg
                stats['cc'] = cc
                stats['gg_loss'] = gg_loss
                stats['gc_loss'] = gc_loss
                stats['cg_loss'] = cg_loss
                stats['cc_loss'] = cc_loss
            elif args.dataset == 'bFFHQ':
                # acc = (stats['avg_acc_group:1'] + stats['avg_acc_group:2'])/2
                acc = torch.Tensor([stats['avg_acc_group:{}'.format(i)] for i in range(4)]).mean()
            if run_wandb:
                for key in stats.keys():
                    metrics['{}_{}'.format(log_mode, key)] = stats[key]
                if args.dataset == 'bFFHQ':
                    metrics['{}_unbiased'.format(log_mode)] = acc    
                else:
                    metrics['{}_worst'.format(log_mode)] = acc
                wandb.log(metrics)
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()
        
            return acc, momentum_grad, group_alpha, optimizer_alpha, stats

def get_stats(n_groups, group_alpha, loss_lambda):
    stats_dict = {}
    for idx in range(n_groups):
        stats_dict[f'alpha_group:{idx}'] = group_alpha[idx].item()
        
    stats_dict['lambda_c'] = loss_lambda.item()
    
    return stats_dict

def train(model, criterion, dataset,
          logger, args, epoch_offset):
    model = model.cuda()
    
    # Curvature-Aware Task Scaling
    if args.dataset == 'MultiCelebA' or args.dataset == 'UrbanCars':
        if '3types' in args.target_name:
            if 'both' in args.grouping_type:
                n_groups = 16
            elif 'total' in args.grouping_type:
                n_groups = 17
            else:
                n_groups = 8
        elif ('bias_group' in args.grouping_type) or ('gc_group' in args.grouping_type):
            n_groups = 4
        else:
            n_groups = 8
    elif args.dataset == 'CelebA':
        if ('bias_group' in args.grouping_type) or ('gc_group' in args.grouping_type):
            n_groups = 2
        else:
            n_groups = 4
    elif args.dataset == 'MultiMNIST':
        if ('bias_group' in args.grouping_type) or ('gc_group' in args.grouping_type):
            n_groups = 4
        else:
            n_groups = 40
    elif args.dataset == 'Waterbirds' or args.dataset == 'bFFHQ':
        if ('bias_group' in args.grouping_type) or ('gc_group' in args.grouping_type):
            n_groups = 2
        else:
            n_groups = 4

    group_alpha = torch.nn.Parameter(torch.Tensor([1/n_groups]*n_groups).cuda())
    loss_lambda = torch.nn.Parameter(torch.Tensor([0]).cuda())
    
    train_loss_computer = LossComputer(
        criterion,
        dataset=dataset['train_data'],
        grouping_type=args.grouping_type,
        dataset_name=args.dataset)

    if args.optimizer=='Adam':
        print("Adam optimizer !!!")
        optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    alpha_lr = args.alpha_lr # 1e-2
    optimizer_alpha = torch.optim.SGD([{'params': group_alpha, 'lr':alpha_lr, 'momentum':0, 'weight_decay':0}])
    optimizer_lambda = torch.optim.SGD([{'params': loss_lambda, 'lr':alpha_lr, 'momentum':0, 'weight_decay':0}])
    
    momentum_grad = None

    best_val_acc = 0
    stop_count = 0
    val_test_worst = 0
    best_val_worst = 0
    best_val_counter = 0

    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        worst, momentum_grad, group_alpha, optimizer_alpha, train_stats = run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, args, log_mode='train',
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            group_alpha=group_alpha, loss_lambda=loss_lambda,
            optimizer_alpha=optimizer_alpha, optimizer_lambda=optimizer_lambda, 
            run_wandb=args.wandb, momentum_grad=momentum_grad
            )

        logger.write(f'\nValidation:\n')
    
        val_loss_computer = LossComputer(
            criterion,
            dataset=dataset['val_data'])
        val_worst, _, _, _, val_stats = run_epoch(
                        epoch, model, optimizer,
                        dataset['val_loader'],
                        val_loss_computer,
                        logger, args, log_mode='val',
                        is_training=False, run_wandb=args.wandb)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                dataset=dataset['test_data'])
            test_worst, _, _, _, test_stats = run_epoch(
                            epoch, model, optimizer,
                            dataset['test_loader'],
                            test_loss_computer,
                            None, args, log_mode ='test',
                            is_training=False, run_wandb=args.wandb)

            if args.dataset == 'bFFHQ':
                ## In case of bFFHQ, 'test_worst' is test unbiased accuracy.
                if best_val_worst <= test_worst:
                    best_val_worst = test_worst
                    val_test_worst = test_worst
                    best_val_counter = 0
                    if args.save_best:
                        torch.save(model, os.path.join(args.log_dir, 'model_epoch_{}.pth'.format(epoch)))
                else:
                    best_val_counter += 1
            else:
                if best_val_worst <= val_worst:
                    best_val_worst = val_worst
                    val_test_worst = test_worst
                    best_val_counter = 0
                    if args.save_best:
                        torch.save(model, os.path.join(args.log_dir, 'model_epoch_{}.pth'.format(epoch)))
                else:
                    best_val_counter += 1

            if best_val_counter > args.early_stop_v: # 10
                print("early stopped !!!")
                break
            if args.wandb:
                wandb.log({'best_val_test_worst': val_test_worst})
                
        logger.write('\n')


def eval(model, criterion, dataset,
          logger, args, epoch_offset):
    model = model.cuda()
    val_test_worst = 0
    best_val_worst = 0
    best_val_counter = 0
    epoch = 0

    logger.write(f'Training: skipped\n')
    logger.write(f'\nValidation: skipped\n')

    logger.write(f'\nTest:\n')

    if dataset['test_data'] is not None:
        test_loss_computer = LossComputer(
            criterion,
            dataset=dataset['test_data'])
        test_worst, _, _, _, test_stats = run_epoch(
                        epoch, model, None,
                        dataset['test_loader'],
                        test_loss_computer,
                        None, args, log_mode ='test',
                        is_training=False, run_wandb=args.wandb)

        # if args.dataset == 'bFFHQ':
        #     if best_val_worst <= test_worst:
        #         best_val_worst = test_worst
        #         val_test_worst = test_worst
        #         best_val_counter = 0
        #     else:
        #         best_val_counter += 1
        # else:
        #     if best_val_worst <= val_worst:
        #         best_val_worst = val_worst
        #         val_test_worst = test_worst
        #         best_val_counter = 0
        #     else:
        #         best_val_counter += 1
        
        # if best_val_counter > args.early_stop_v: # 10
        #     print("early stopped !!!")
        #     break
        # if args.wandb:
            # wandb.log({'best_val_test_worst': val_test_worst})
    group_counts = dataset['train_data'].group_counts()
    if args.dataset == 'MultiCelebA' or args.dataset == 'UrbanCars':
        if '3types' in args.target_name:
            n_groups = 16
        else:
            n_groups = 8
    elif args.dataset == 'MultiMNIST':
        n_groups = 40
    elif args.dataset == 'Waterbirds' or args.dataset == 'CelebA' or args.dataset == 'bFFHQ':
        n_groups = 4
    test_indist_acc = (torch.Tensor([test_stats['avg_acc_group:{}'.format(i)] * group_counts[i] for i in range(n_groups)])/ group_counts.sum()).sum()
    test_unbiased_acc = torch.Tensor([test_stats['avg_acc_group:{}'.format(i)] for i in range(n_groups)]).mean()
    if args.dataset == 'MultiCelebA' or args.dataset == 'MultiMNIST':
        if '3types' in args.target_name:
            test_ccc = (test_stats['avg_acc_group:5'] + test_stats['avg_acc_group:10'])/2
            logger.write('\nCCC: {:.2f} Unbiased: {:.2f} InDist: {:.2f} \n'.format(test_ccc*100, test_unbiased_acc*100, test_indist_acc*100))
        else:
            logger.write('\nGG: {:.2f} GC: {:.2f} CG: {:.2f} CC: {:.2f} Unbiased: {:.2f} InDist: {:.2f} \n'.format(test_stats['gg']*100, test_stats['gc']*100, test_stats['cg']*100, test_stats['cc']*100, test_unbiased_acc*100, test_indist_acc*100))
    elif args.dataset == 'UrbanCars':
        test_cc = (test_stats['avg_acc_group:3'] + test_stats['avg_acc_group:4'])/2
        logger.write('\n InDist: {:.2f} CC: {:.2f} Gap: {:.2f} \n'.format(test_indist_acc*100, test_cc*100, (test_indist_acc-test_cc)*100))
    elif args.dataset == 'Waterbirds' or args.dataset == 'CelebA':
        logger.write('\n InDist: {:.2f} Worst: {:.2f} \n'.format(test_indist_acc*100, test_worst*100))
    elif args.dataset == 'bFFHQ':
        logger.write('\n Unbiased: {:.2f} \n'.format(test_worst*100))
    logger.write('\n')
    