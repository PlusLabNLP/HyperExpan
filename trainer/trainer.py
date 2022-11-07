import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from base import BaseTrainer
import dgl
from tqdm import tqdm
import time, copy, random
import itertools
import json
import more_itertools as mit
from functools import partial
from collections import defaultdict
from model.model import *
from model.loss import *
from data_loader.data_loaders import *
from model.metric import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
import seaborn as sns


MAX_CANDIDATE_NUM=100000


def rearrange(energy_scores, candidate_position_idx, true_position_idx, 
                print_breakdown=False, vocab=None, logger=None):
    """
    Arguments:
        energy_scores (1-d tensor): energy scores for every candidate, length example: 11938
        candidate_position_idx (list): word idx for candidates aligned with energ_scores list, length example: 11938
        true_position_idx: true position's word idx
    """
    tmp = np.array([[x==y for x in candidate_position_idx] for y in true_position_idx]).any(0)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()

    # print analysis
    if print_breakdown:
        logger.info(f'neighbors: {[vocab[idx] for idx in true_position_idx]} \t energy: {energy_scores[correct].detach().cpu().numpy()}')

        dists = energy_scores.squeeze()
        sorted_dists, sorted_idx = dists.sort() 
        sorted_idx = list(sorted_idx.detach().cpu().numpy())
        sorted_dists = list(sorted_dists.detach().cpu().numpy())
        # if larger value is better
        sorted_idx.reverse()
        sorted_dists.reverse()
        logger.info(sorted_dists[0:20])
        sorted_words = [vocab[idx] for idx in sorted_idx]
        logger.info(sorted_words[0:20])

        # calcualte ranks of ground truth neighbours
        # get order idx in energy_scores for true_position_idx
        true_position_idx_in_energy_list = [candidate_position_idx.index(y) for y in true_position_idx]
        ranks, = np.where(np.in1d(sorted_idx, list(true_position_idx_in_energy_list)))
        logger.info(ranks)
    
    energy_scores = torch.cat((energy_scores[correct], energy_scores[incorrect]))
    return energy_scores, labels

def score2pred(scores):
    """
    Arguments:
        scores: tensor of size [batch_size, 1]
    Return:
        Numpy list of prediction of size [batch_size]
    """
    return np.round(torch.sigmoid(scores).squeeze().detach().cpu().numpy())

def energy2pred(energy_scores, labels, threshold=0.5):
    """
    Arguments:
        energy_scores: tensors of size [batch_size, 1], energy scores, smaller means closer to correct
    Return:
        numpy list of labels, if the energy score is smaller than 0.5 it's correct, so the item in the list is 1
    """
    reverse_dict = {
        1: 0,
        0: 1
    }
    scores = torch.sigmoid(energy_scores).squeeze().detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    pred = [labels[i] if s < threshold else reverse_dict[labels[i]] for i, s in enumerate(scores)]
    return pred


def get_wn_id_from_str(node_str):
    if node_str in ['root', 'leaf']:
        return node_str
    else:
        return node_str.split('||')[1].split('@@@')[0]

def gen_plot(input_list):
    """Create a pyplot plot and save to buffer."""
    plt.figure()

    sns.distplot(input_list, hist=True, kde=True, 
             color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.test_batch_size = config['trainer']['test_batch_size']
        self.is_infonce_training = config['loss'].startswith("info_nce")
        self.is_focal_loss = config['loss'].startswith("FocalLoss")
        self.data_loader = data_loader
        self.do_validation = True
        self.do_validation_test = True # whether perform test after eval_test_each epoch
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_mode = self.config['lr_scheduler']['args']['mode']  # "min" or "max"
        self.log_step = len(data_loader) // 5
        self.pre_metric = pre_metric

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        all_ranks = self.pre_metric(output, target)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(all_ranks)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def print_cls_report(self, labels, pred):
        confusion_matrix_print = confusion_matrix(labels, pred)
        clas_report = cls_report(pred, labels, output_dict=False)
        self.logger.info(confusion_matrix_print)
        self.logger.info(clas_report)

    def test(self, epoch=-1):
        test_return = self._test('test')
        # overall metrics for all query nodes
        test_values = test_return[0]
        for i, mtr in enumerate(self.metrics):
            self.logger.info('    {:15s}: {:.3f}'.format('test_' + mtr.__name__, test_values[i]))
        
        # save testing result to tensorboard
        if epoch >= 0:
            test_log_dict = {}
            for i, metric_name in enumerate(self.config['metrics']):
                test_log_dict[metric_name] = test_values[i]
            self.writer.add_msg_dict(test_log_dict, epoch, prefix="test")
        return test_values


class TrainerExpan(Trainer):
    """
    Trainer class, for one-to-one matching methods on taxonomy expansion task

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerExpan, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler)
        self.mode = mode

        dataset = self.data_loader.dataset
        self.vocab = dataset.vocab
        self.vocab_level = dataset.vocab_level
        self.vocab_level_max = dataset.max_level
        # current implementation will use all_nodes by default
        self.candidate_positions = dataset.all_nodes
        self.logger.info(f'TrainerExpan candidate_positions: {len(self.candidate_positions)}')
        self.valid_node2pos = dataset.valid_node2pos
        self.test_node2pos = dataset.test_node2pos
        self.valid_vocab = dataset.valid_node_list
        self.test_vocab = dataset.test_node_list

        self.train_step_overall = 0

        if 'g' in mode:
            self.all_nodes = sorted(list(dataset.core_subgraph.nodes))
            self.node2subgraph = {node: dataset._get_subgraph_and_node_pair(-1, node) for node in tqdm(self.all_nodes, desc='collecting nodegraph')}

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        labels_all = []
        loss_all = []
        loss_for_each_instances = False
        if 'burnin' in self.config['optimizer']['args']: 
            # optimizer for hyperbolic training init lr is burnin lr
            if epoch == self.config['optimizer']['args']['burnin']:
                # change lr to normal lr after burnin epochs
                self.optimizer.param_groups[0]["lr"] = self.config['optimizer']['args']['lr']
        lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(f'lr={lr}')

        if epoch <= 1:
            # do a validation before training start
            _ = self._test_container(0)

        for batch_idx, batch in enumerate(tqdm(self.data_loader)):
            nf, label, u, graphs, paths, lens = batch
            # nf: queries tensor; u: parent mounting points

            self.optimizer.zero_grad()
            label = label.to(self.device)

            scores = self.model(nf, u, graphs, paths, lens)

            if self.is_infonce_training or self.model.options['matching_method'] == 'PE':
                n_batches = label.sum().detach()
                target = torch.zeros(n_batches, dtype=torch.long).to(self.device)
                prediction = scores.reshape(n_batches, -1)
                loss = self.loss(prediction, target)
            else:
                loss_for_each_instances = True
                loss, loss_each = self.loss(scores, label, return_none_reduction=True)

            loss.backward()
            if 'max_grad_norm' in self.config['optimizer']['args']:
                if self.config['optimizer']['args']['max_grad_norm'] >= 0:
                    clip_grad_norm_(self.model.parameters(), self.config['optimizer']['args']['max_grad_norm'])
            self.optimizer.step()

            total_loss += loss.item()
            if loss_for_each_instances:
                labels_all += list(label.detach().cpu().numpy())
                loss_all += list(loss_each.detach().cpu().numpy())
            self.writer.add_msg_dict({'loss-by-step': loss.item()}, self.train_step_overall, prefix="train")

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'
                        .format(epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                                100.0 * batch_idx / len(self.data_loader),
                                loss.item()))
            self.train_step_overall += 1

        # print learned curvature this round
        if 'propagation_method' in self.model.options:
            if self.model.options['propagation_method'] == 'HGCN':
                self.logger.info(f'Curvatures: {self.model.parent_graph_propagate.curvatures}')

        # print learned readout weight
        if 'readout_method' in self.model.options:
            if self.model.options['readout_method'] in ['MR1', 'WMR']:
                self.logger.info(f"Readout weights: {self.model.p_readout.weights.weight.squeeze()}")

        if loss_for_each_instances:
            # Calculate loss by category
            loss_0 = [r for i, r in enumerate(loss_all) if labels_all[i] == 0]
            loss_1 = [r for i, r in enumerate(loss_all) if labels_all[i] == 1]

        log = {'loss': total_loss / len(self.data_loader),
                'lr': lr}
        if loss_for_each_instances:
            log['loss_0'] = np.mean(loss_0)
            log['loss_1'] = np.mean(loss_1)
        self.writer.add_msg_dict(log, epoch, prefix="train")

        ## Validation stage
        if self.do_validation and (epoch % self.eval_each == 0):
            val_log = self._test_container(epoch)
            log = {**log, **val_log}

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if self.lr_scheduler_mode == "min":
                        self.lr_scheduler.step(log['val_metrics'][0]) # macro_mr normally
                    else:
                        self.lr_scheduler.step(log['val_metrics'][-1]) # mrr_scaled_10 normally
                else:
                    self.lr_scheduler.step()

        if self.do_validation_test and (epoch % self.eval_test_each == 0):
            _ = self.test(epoch=epoch)

        return log

    def _test_container(self, epoch):
        """
        A function to call _test and process the returned results
        """
        test_result = self._test('validation')
        val_log = {'val_metrics': test_result[0][0],
                   'val_metrics_min': test_result[0][1],
                   'val_metrics_max': test_result[0][2]}

        val_log_dict = test_result[4] # take other metrics as base to save to tensorboard
        # Convert val result to format that can write into tensorboard
        for i, metric_name in enumerate(self.config['metrics']):
            val_log_dict[metric_name] = val_log['val_metrics'][i]
            val_log_dict['min_%s' % metric_name] = val_log['val_metrics_min'][i]
            val_log_dict['max_%s' % metric_name] = val_log['val_metrics_max'][i]
        self.writer.add_msg_dict(val_log_dict, epoch, prefix="val")

        if epoch == 0:
            # print log information to the screen
            self._result2log(val_log, epoch)
        return val_log

    def test(self, epoch=-1):
        test_return = self._test('test', by_level=True)
        # overall metrics for all query nodes
        test_values = test_return[0][0]
        for i, mtr in enumerate(self.metrics):
            self.logger.info('    {:15s}: {:.3f}'.format('test_' + mtr.__name__, test_values[i]))
        
        # metrics eval for each level
        metrics_all_levels_dict = test_return[5]
        level_numbers = []
        count_across_levels = []
        mrr_across_levels = []
        hit10_across_levels = []
        mr_across_levels = []
        for level_number, metrics_dict in metrics_all_levels_dict.items():
            level_numbers.append(level_number)
            count_across_levels.append(metrics_dict['count'])
            mrr_across_levels.append(metrics_dict['metrics'][-1])
            hit10_across_levels.append(metrics_dict['metrics'][5])
            mr_across_levels.append(metrics_dict['metrics'][0])
            # for i, mtr in enumerate(self.metrics):
            #     self.logger.info('    {:15s}: {:.3f}'.format('test_' + mtr.__name__, test_values[i]))
        self.logger.info(f'across levels: {level_numbers}')
        self.logger.info(f'across levels count: {count_across_levels}')
        self.logger.info(f'across levels mrr_scaled_10: {mrr_across_levels}')
        self.logger.info(f'across levels hit_at_10: {hit10_across_levels}')
        self.logger.info(f'across levels macro_mr: {mr_across_levels}')
        
        # save testing result to tensorboard
        if epoch >= 0:
            test_log_dict = {}
            for i, metric_name in enumerate(self.config['metrics']):
                test_log_dict[metric_name] = test_values[i]
            self.writer.add_msg_dict(test_log_dict, epoch, prefix="test")
        return test_values
    
    def _test(self, mode, gpu=True, by_level=False):
        assert mode in ['test', 'validation']
        torch.cuda.empty_cache()
        model = self.model if gpu else self.model.cpu()

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            node_features = dataset.node_features
            if mode == 'test':
                node2pos = dataset.test_node2parent
                vocab = list(node2pos.keys())
            else:
                vocab = self.valid_vocab
                node2pos = self.valid_node2pos
            candidate_positions = self.candidate_positions
            batched_model = [] # save the CPU graph representation
            batched_positions = []
            for us_l in tqdm(mit.sliced(candidate_positions, batch_size), desc="Generating graph encoding ..."):

                bgu, bpu, lens = None, None, None
                if 'r' in self.mode or 'l' in self.mode:
                    us = torch.tensor(us_l)
                if 'g' in self.mode:
                    bgu = [self.node2subgraph[e] for e in us_l]
                if 'p' in self.mode:
                    bpu, lens = dataset._get_batch_edge_node_path(us_l)
                    bpu = bpu
                    lens = lens
                ur = self.model.forward_encoders(us, bgu, bpu, lens)
                batched_model.append(ur.detach().cpu())
                batched_positions.append(len(us))

            # start per query prediction
            all_ranks = []
            all_ranks_min = []
            all_ranks_max = []
            all_num_same_val = []
            self.logger.info(f'length of candidate positions: {len(candidate_positions)}')
            if self.eval_topn > 0:
                vocab = list(vocab)[:self.eval_topn]
            self.logger.info(f'length of objects: {len(vocab)}') #1000
            for i, query in tqdm(enumerate(vocab), desc=mode, total=len(vocab)):
                batched_energy_scores = []
                if 'encoder_query' in self.model.options:
                    nf = model.forward_encoders_query(torch.tensor([query]))
                elif 'l' in self.mode:
                    nf = model.forward_encoders(torch.tensor([query]), None, None, None)
                else:
                    nf = node_features[query, :].to(self.device)
                for ur, n_position in zip(batched_model, batched_positions):
                    expanded_nf = nf.expand(n_position, -1)
                    ur = ur.to(self.device)
                    energy_scores = model.match(ur, expanded_nf)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores) # [11938]
                print_breakdown = False
                if print_breakdown:
                    self.logger.info('======')
                    self.logger.info(f'node: {self.vocab[query]}')
                batched_energy_scores, labels = rearrange(batched_energy_scores, candidate_positions, node2pos[query],
                                                            print_breakdown=print_breakdown, vocab=self.vocab, 
                                                            logger=self.logger)
                tmp_ranks, tmp_ranks_min, tmp_ranks_max, tmp_num_same_val = self.pre_metric(batched_energy_scores, labels,
                                                                                            return_rank_range=True,
                                                                                            print_breakdown=print_breakdown)
                if print_breakdown:
                    self.logger.info(f'ranks: {tmp_ranks} {tmp_ranks_min} {tmp_ranks_max} {tmp_num_same_val}')
                all_ranks.extend(tmp_ranks)
                all_ranks_min.extend(tmp_ranks_min)
                all_ranks_max.extend(tmp_ranks_max)
                all_num_same_val.extend(tmp_num_same_val)
            metrics_avg = [metric(all_ranks) for metric in self.metrics]
            metrics_min = [metric(all_ranks_min) for metric in self.metrics]
            metrics_max = [metric(all_ranks_max) for metric in self.metrics]
            total_metrics = [metrics_avg, metrics_min, metrics_max]
            # return other_metrics_dict to provide num_same_val information
            mean_num_same_val = np.array([np.array(all_rank).mean() for all_rank in all_num_same_val]).mean()
            other_metrics_dict = {'mean_num_same_val': mean_num_same_val}

            metrics_in = []
            metrics_out = []

            # get rank result for each level
            metrics_all_levels_dict = {}
            if by_level:
                for level_this in range(self.vocab_level_max + 1):
                    all_ranks_this_level = [r for i, r in enumerate(all_ranks) if self.vocab_level[i] == level_this]
                    metrics_this_level = [metric(all_ranks_this_level) for metric in self.metrics]
                    metrics_all_levels_dict[level_this] = {
                        "count": len(all_ranks_this_level),
                        "metrics": metrics_this_level
                    }

            emb_all_words = None
        return total_metrics, metrics_in, metrics_out, emb_all_words, other_metrics_dict, metrics_all_levels_dict