import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.optimizer as module_optimizer
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import trainer as trainer_arch
from functools import partial
import time
import logging
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from operator import itemgetter
import random
from prettytable import PrettyTable
import geoopt as gt

def count_parameters(model, logger):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    logger.info(table)
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params

#Get the surface form word (i.e. breathe) from WordNet term
#lemma level id: breath.v.01.respire, synset id: breathe.v.01
def get_verb_from_id(wordid, source='wn', remove_under=True, return_single_word=False):
    # get_word_from_class_name or wordnet or verbnet
    result = ''
    if '.' in wordid or '-' in wordid:
        if source == 'wn':
            if len(wordid.split('.')) > 3:
                # when it's lemma level id, like: breathe.v.01.respire
                result = wordid.split('.')[-1]
            else:
                # when it's synset id, like: breathe.v.01
                result = wordid.split('.')[0]

            result = result.replace('-', ' ')
            
            if return_single_word:
                # return `force` for `force_out`
                result = result.split('_')[0]

            if remove_under:
                result = result.replace('_', ' ')
        elif source == 'vn':
            result = wordid.split('-')[0]
            result = result.replace('_', ' ')
    else:
        result = wordid
    return result

#give the name as a string with extension (i.e. name.png)
def plot_multimean_hist(freq, name):
    bins_list = list(range(1, max(freq) + 2))
    fig, ax = plt.subplots()
    if name == "train_meanings.png" or name == "vocab_meanings.png" or ("_total" in name):  
        fig, ax = plt.subplots(figsize=(30, 8))
    counts, bins, patches = ax.hist(freq, bins=bins_list)
    #Ticks are at edges of the bins
    ax.set_xticks(bins)

    #Labels
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        #Label the raw counts
        ax.annotate(str(int(count)), xy=(x, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -18), textcoords="offset points", va="top", ha="center")
        #Label the percentages
        percent = "%0.2f%%" % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -32), textcoords="offset points", va="top", ha="center")

    #More room at the bottom of the plot
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("Number of Meanings for Surface Form", labelpad=23)
    plt.ylabel("Frequency")
    plt.savefig("data/meanings/" + name, dpi=300)
    plt.close()

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_data_loader = config.initialize('train_data_loader', module_data, config['mode'], config['data_path'])
    logger.info(train_data_loader)
    
    # build model architecture, then print to console
    node_features = train_data_loader.dataset.node_features
    vocab_size, embed_dim = node_features.size()
    if config['arch']['type'] in ['ExpanMatchLMModel', 'MatchLMModel']:
        word2sen = train_data_loader.word2sen
        logger.info('preapred word2sen from train_data_loader')
    if config['arch']['type'] == 'ExpanMatchPEModel':
        # add _vocab_size attribute in the config json for PE network initialization
        config['arch']['args']['_vocab_size'] = vocab_size
    
    # initialize model
    if config['arch']['type'] not in ['ExpanMatchLMModel', 'ExpanMatchPEModel']:
        model = config.initialize('arch', module_arch, config['mode'])
    else:
        model = config.initialize('arch', module_arch, config['mode'], vocab=train_data_loader.vocab)
    
    if config['arch']['type'] in ['ExpanMatchLMModel', 'MatchLMModel']:
        model.set_word2sen(word2sen)
    elif config['arch']['type'] != 'ExpanMatchPEModel':
        if config['arch']['args']['pretrained_embedding']:
            # use pretrained embedding
            model.set_embedding(vocab_size=vocab_size, embed_dim=embed_dim, pretrained_embedding=node_features,
                                freeze=config['arch']['args']['embedding_freeze'])
        else:
            # use random embedding
            model.set_embedding(vocab_size=vocab_size, embed_dim=embed_dim, 
                                freeze=config['arch']['args']['embedding_freeze'])
    logger.info(model)

    # print number of parameters
    count_parameters(model, logger)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    if config['loss'].startswith("FocalLoss"):
        loss = loss()
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    if config['loss'].startswith("info_nce") or \
       config['loss'].startswith("bce_loss"):
        pre_metric = partial(module_metric.obtain_ranks, mode=1)  # info_nce_loss
    else:
        pre_metric = partial(module_metric.obtain_ranks, mode=0)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if 'burnin' in config['optimizer']['args']:
        init_lr = config['optimizer']['args']['lr'] * config['optimizer']['args']['burnin_multiplier']
    else:
        init_lr = config['optimizer']['args']['lr']
    # use existing optimizer defined in torch.optim
    if config['optimizer']['type'] == 'RAdam' or config['optimizer']['type'] == 'RiemannianAdam':
        from hype.radam import RiemannianAdam
        optimizer = RiemannianAdam(trainable_params,
                                    lr=init_lr,
                                    stabilize=5)
    else:
        optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    start = time.time()
    Trainer = config.initialize_trainer('arch', trainer_arch)
    trainer = Trainer(config['mode'], model, loss, metrics, pre_metric, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      lr_scheduler=lr_scheduler)
    evaluations = trainer.train()
    end = time.time()
    logger.info(f"Finish training in {end-start} seconds")
    return evaluations


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Training taxonomy expansion model')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--suffix', default="", type=str, help='suffix indicating this run (default: None)')
    args.add_argument('-n', '--n_trials', default=1, type=int, help='number of trials (default: 1)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        # Data loader (self-supervision generation)
        CustomArgs(['--train_data'], type=str, target=('train_data_loader', 'args', 'data_path')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('train_data_loader', 'args', 'batch_size')),
        CustomArgs(['--ns', '--negative_size'], type=int, target=('train_data_loader', 'args', 'negative_size')),
        CustomArgs(['--ef', '--expand_factor'], type=int, target=('train_data_loader', 'args', 'expand_factor')),
        CustomArgs(['--crt', '--cache_refresh_time'], type=int, target=('train_data_loader', 'args', 'cache_refresh_time')),
        CustomArgs(['--nw', '--num_workers'], type=int, target=('train_data_loader', 'args', 'num_workers')),
        CustomArgs(['--sm', '--sampling_mode'], type=int, target=('train_data_loader', 'args', 'sampling_mode')),
        # Trainer & Optimizer
        CustomArgs(['--mode'], type=str, target=('mode', )),
        CustomArgs(['--loss'], type=str, target=('loss', )),
        CustomArgs(['--ep', '--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--es', '--early_stop'], type=int, target=('trainer', 'early_stop')),
        CustomArgs(['--tbs', '--test_batch_size'], type=int, target=('trainer', 'test_batch_size')),
        CustomArgs(['--v', '--verbose_level'], type=int, target=('trainer', 'verbosity')),
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--wd', '--weight_decay'], type=float, target=('optimizer', 'args', 'weight_decay')),
        CustomArgs(['--l1'], type=float, target=('trainer', 'l1')),
        CustomArgs(['--l2'], type=float, target=('trainer', 'l2')),
        CustomArgs(['--l3'], type=float, target=('trainer', 'l3')),
        # Model architecture
        CustomArgs(['--pm', '--propagation_method'], type=str, target=('arch', 'args', 'propagation_method')),
        CustomArgs(['--rm', '--readout_method'], type=str, target=('arch', 'args', 'readout_method')),
        CustomArgs(['--mm', '--matching_method'], type=str, target=('arch', 'args', 'matching_method')),
        CustomArgs(['--k'], type=int, target=('arch', 'args', 'k')),
        CustomArgs(['--in_dim'], type=int, target=('arch', 'args', 'in_dim')),
        CustomArgs(['--hidden_dim'], type=int, target=('arch', 'args', 'hidden_dim')),
        CustomArgs(['--out_dim'], type=int, target=('arch', 'args', 'out_dim')),
        CustomArgs(['--pos_dim'], type=int, target=('arch', 'args', 'pos_dim')),
        CustomArgs(['--num_heads'], type=int, target=('arch', 'args', 'heads', 0)),
        CustomArgs(['--feat_drop'], type=float, target=('arch', 'args', 'feat_drop')),
        CustomArgs(['--attn_drop'], type=float, target=('arch', 'args', 'attn_drop')),
        CustomArgs(['--hidden_drop'], type=float, target=('arch', 'args', 'hidden_drop')),
        CustomArgs(['--out_drop'], type=float, target=('arch', 'args', 'out_drop')),
    ]
    config = ConfigParser(args, options)
    args = args.parse_args()
    n_trials = args.n_trials

    if n_trials > 0:
        config.get_logger('train').info(f'number of trials: {n_trials}')
        metrics = config['metrics']
        save_file = config.log_dir / 'evaluations.txt'
        fin = open(save_file, 'w')
        fin.write('\t'.join(metrics))

        evaluations = []
        for i in range(n_trials):
            config.set_save_dir(i+1)
            res = main(config)
            evaluations.append(res)
            fin.write('\t'.join([f'{i:.3f}' for i in res]))

        evaluations = np.array(evaluations)
        means = evaluations.mean(axis=0)
        stds = evaluations.std(axis=0)
        final_output = '  '.join([f'& {i:.3f} +- {j:.3f}' for i, j in zip(means, stds)])
        fin.write(final_output)
        config.get_logger('train').info(final_output)
    else:
        main(config)
