import torch
import torch.nn.functional as F
import numpy as np
import copy

class LossComputer:
    def __init__(self, criterion, dataset, gamma=0.1, adj=None, dataset_name='MultiCelebA', grouping_type='gc_group'):
        self.criterion = criterion
        self.gamma = gamma
        self.grouping_type = grouping_type
        self.n_groups = dataset.n_groups
        
        self.group_counts = dataset.group_counts().cuda()
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_str = dataset.group_str

        # quantities maintained throughout training
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False, 
                optimizer=None, group_alpha=None, loss_lambda=None, metrics=None, momentum_grad=None, grad_log=False, alpha_step=0):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        
        self.update_exp_avg_loss(group_loss, group_count)
        
        # compute overall loss
        if is_training:
            if 'gc_group' in self.grouping_type:
                if len(group_loss) == 16:
                    group_mean = group_count>0
                    # cats_group_loss = torch.stack([(group_loss[i]+group_loss[15-i])/(group_mean[i]+group_mean[15-i]+1e-8) for i in range(8)])
                    cats_group_loss = torch.stack([(group_loss[i]+group_loss[15-i])/2 for i in range(8)])
                elif len(group_loss) == 8:
                    group_mean = group_count>0
                    cats_group_loss = torch.stack([(group_loss[i]+group_loss[7-i])/(group_mean[i]+group_mean[7-i]+1e-8) for i in range(4)])
                elif len(group_loss) == 4: # waterbirds
                    group_mean = group_count>0
                    cats_group_loss = torch.stack([(group_loss[i]+group_loss[3-i])/(group_mean[i]+group_mean[3-i]+1e-8) for i in range(2)])
                elif len(group_loss) == 40:
                    group_mean = group_count>0
                    idx = [np.array([*range(10)])*4+k for k in range(4)]
                    cats_group_loss = torch.stack([(group_loss[idx[i]].sum())/(group_mean[idx[i]].sum()+1e-8) for i in range(4)])
                elif len(group_loss) == 100:
                    group_mean = group_count>0
                    G_idx = [k*10+k for k in range(10)]
                    total_idx = [*range(100)]
                    C_idx = [x for x in total_idx if x not in G_idx]
                    
                    cats_group_loss = torch.stack([(group_loss[G_idx].sum())/(group_mean[G_idx].sum()+1e-8), (group_loss[C_idx].sum())/(group_mean[C_idx].sum()+1e-8)])
                
            elif 'bias_group' in self.grouping_type:
                if len(group_loss) == 16:
                    group_mean = group_count>0
                    # cats_group_loss = torch.stack([(group_loss[i]+group_loss[15-i])/(group_mean[i]+group_mean[15-i]+1e-8) for i in range(8)])
                    cats_group_loss = torch.stack([(group_loss[i]+group_loss[i+8])/2 for i in range(8)])
                elif len(group_loss) == 8:
                    group_mean = group_count>0
                    cats_group_loss = torch.stack([(group_loss[i]+group_loss[i+4])/(group_mean[i]+group_mean[i+4]+1e-8) for i in range(4)])
                elif len(group_loss) == 4:
                    group_mean = group_count>0
                    cats_group_loss = torch.stack([(group_loss[i]+group_loss[i+2])/(group_mean[i]+group_mean[i+2]+1e-8) for i in range(2)])
                elif len(group_loss) == 40:
                    group_mean = group_count>0
                    idx = [np.array([*range(10)])*4+k for k in range(4)]
                    cats_group_loss = torch.stack([(group_loss[idx[i]].sum())/(group_mean[idx[i]].sum()+1e-8) for i in range(4)])
            else:
                cats_group_loss = group_loss

            if alpha_step:
                for i in range(cats_group_loss.shape[0]):
                    metrics['group_loss:{}'.format(i)] = cats_group_loss[i]

                pure_grads, shapes, _ = self._pack_grad(cats_group_loss, optimizer)
                pure_grads = torch.stack(pure_grads)
                
                new_grad = (pure_grads * (F.softmax(group_alpha.detach(), dim=0).reshape(-1,1))).sum(dim=0)
                new_grad = self._unflatten_grad(new_grad, shapes[0])
                
                self._set_grad(new_grad, optimizer)
                
                grads = copy.deepcopy(pure_grads)

                
                first_term = (cats_group_loss.detach()*F.softmax(group_alpha, dim=0)).sum()
                metrics['first_term'] = first_term.cpu().data
                
                hessian_approx = (((F.softmax(group_alpha, dim=0).reshape(-1,1)*grads).sum(dim=0).norm(2))**2).sum()
                metrics['hessian_approx'] = hessian_approx.cpu().data
                hessian_approx = loss_lambda*hessian_approx
                
                actual_loss = first_term+hessian_approx
                metrics['actual_loss'] = actual_loss.cpu().data
                
                weights = None
            else:
                weights = None
                actual_loss = (cats_group_loss * F.softmax(group_alpha.detach(), dim=0)).sum()
        else:
            actual_loss = per_sample_losses.mean()
            weights = None
            
        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)
        return actual_loss, metrics, momentum_grad

    def _set_grad(self, grads, optimizer):
        '''
        set the modified gradients to the network
        '''
        idx = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return
    
    def _pack_grad(self, objectives, optimizer):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        # import time
        grads, shapes, has_grads = [], [], []
        # start_time = time.time()
        objectives[0].backward(retain_graph=True)
        # print(time.time()-start_time)
        # import pdb; pdb.set_trace()
        # times = 0
        for counter, obj in enumerate(objectives):
            # import pdb; pdb.set_trace()
            # start_pack = time.time()
            optimizer.zero_grad(set_to_none=True)
            if counter == len(objectives)-1:
                obj.backward()
            else:
                obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad(optimizer)
            # iter_time = time.time()-start_pack
            # print("pack", iter_time)
            # times += iter_time
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        # print("average: ", times/len(objectives))
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self, optimizer):
            grad, shape, has_grad = [], [], []
            for group in optimizer.param_groups:
                for p in group['params']:
                    # if p.grad is None: continue
                    # tackle the multi-head scenario
                    if p.grad is None:
                        shape.append(p.shape)
                        grad.append(torch.zeros_like(p).to(p.device))
                        has_grad.append(torch.zeros_like(p).to(p.device))
                        continue
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p).to(p.device))
            return grad, shape, has_grad

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        self.update_data_counts += group_count
        self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            # stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
        
            # stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            # stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            # stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        # stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        # stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        # stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        # logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        # logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        # logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')

        ####  #### #### #### #### ####For Multi MNIST #### #### #### #### #### #### ####
        # index = np.array([*range(10)])*4
        # logger.write(f'Average gg-acc: {self.avg_group_acc[index].mean().item():.3f}  \n') # gg
        # index = np.array([*range(10)])*4+1
        # logger.write(f'Average gc-acc: {self.avg_group_acc[index].mean().item():.3f}  \n') # gc
        # index = np.array([*range(10)])*4+2
        # logger.write(f'Average cg-acc: {self.avg_group_acc[index].mean().item():.3f}  \n') # cg
        # index = np.array([*range(10)])*4+3
        # logger.write(f'Average cc-acc: {self.avg_group_acc[index].mean().item():.3f}  \n') # cc
        #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        
        
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()

