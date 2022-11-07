import torch
import numpy as np
import itertools
import re
import math
from sklearn.metrics import classification_report


def acc(outputs, targets):
    prediction = outputs.cpu().numpy().squeeze() > 0
    label = targets.cpu().numpy()
    acc = (prediction==label).sum() / len(prediction)
    return acc

def cls_report(outputs, labels, output_dict=True):
    cls_report_dict = classification_report(labels, outputs, output_dict=output_dict)
    return cls_report_dict

def calculate_ranks_from_similarities(all_similarities, positive_relations, print_breakdown=False):
    """
    all_similarities: a np array
    positive_relations: a list of array indices

    return a list
    """
    ranks = list(np.argsort(np.argsort(-all_similarities))[positive_relations])#+1)
    
    # add on procedure, if many similaries have exact same value
    # since the index list is sorted, so positive_relations which are normally very first item with index 0
    # should be have smallest rank among all same similarities idx.
    # return three ranks to indicate the range of possible ranks according to different way to calculate
    ranks_min = ranks
    ranks_max = [0]*len(ranks)
    ranks_avg = [0]*len(ranks)
    num_same_val = [0]*len(ranks)
    for i, r in enumerate(ranks):
        pos_idx = positive_relations[i]
        this_similarity_value = all_similarities[pos_idx]
        same_similarity_count = list(all_similarities).count(this_similarity_value)
        ranks_avg[i] = ranks[i] + math.ceil(same_similarity_count/2)
        ranks_max[i] = ranks[i] + same_similarity_count
        num_same_val[i] = same_similarity_count
    return ranks_avg, ranks_min, ranks_max, num_same_val

def calculate_ranks_from_distance(all_distances, positive_relations, print_breakdown=False):
    """
    all_distances: a np array
    positive_relations: a list of array indices

    return a list
    """
    ranks = list(np.argsort(np.argsort(all_distances))[positive_relations]+1)

    # add on procedure, if many similaries have exact same value
    # since the index list is sorted, so positive_relations which are normally very first item with index 0
    # should be have smallest rank among all same similarities idx.
    # return three ranks to indicate the range of possible ranks according to different way to calculate
    ranks_min = ranks
    ranks_max = [0]*len(ranks)
    ranks_avg = [0]*len(ranks)
    num_same_val = [0]*len(ranks)
    for i, r in enumerate(ranks):
        pos_idx = positive_relations[i]
        this_similarity_value = all_distances[pos_idx]
        same_similarity_count = list(all_distances).count(this_similarity_value)
        ranks_avg[i] = ranks[i] + math.ceil(same_similarity_count/2)
        ranks_max[i] = ranks[i] + same_similarity_count
        num_same_val[i] = same_similarity_count
    return ranks, ranks_min, ranks_max, num_same_val

def obtain_ranks(outputs, targets, mode=0, return_rank_range=False, print_breakdown=False):
    """ 
    outputs : tensor of size (batch_size, 1), required_grad = False, model predictions
    targets : tensor of size (batch_size, ), required_grad = False, labels
        Assume to be of format [1, 0, ..., 0, 1, 0, ..., 0, ..., 0]
    mode == 0: rank from distance (smaller is preferred)
    mode == 1: rank from similarity (larger is preferred)
    """
    if mode == 0:
        calculate_ranks = calculate_ranks_from_distance
    else:
        calculate_ranks = calculate_ranks_from_similarities
    all_ranks = []
    all_ranks_min = []
    all_ranks_max = []
    all_num_same_val = []
    prediction = outputs.cpu().numpy().squeeze()
    label = targets.cpu().numpy()
    sep = np.array([0, 1], dtype=label.dtype)
    
    # fast way to find subarray indices in a large array, c.f. https://stackoverflow.com/questions/14890216/return-the-indexes-of-a-sub-array-in-an-array
    end_indices = [(m.start() // label.itemsize)+1 for m in re.finditer(sep.tostring(), label.tostring())]
    end_indices.append(len(label)+1)
    start_indices = [0] + end_indices[:-1]
    for start_idx, end_idx in zip(start_indices, end_indices):
        distances = prediction[start_idx: end_idx]
        labels = label[start_idx:end_idx]
        positive_relations = list(np.where(labels == 1)[0])
        ranks, ranks_min, ranks_max, num_same_val = calculate_ranks(distances, positive_relations, print_breakdown=print_breakdown)
        all_ranks.append(ranks)
        all_ranks_min.append(ranks_min)
        all_ranks_max.append(ranks_max)
        all_num_same_val.append(num_same_val)
    if return_rank_range:
        return all_ranks, all_ranks_min, all_ranks_max, all_num_same_val
    else:
        return all_ranks

def macro_mr(all_ranks):
    macro_mr = np.array([np.array(all_rank).mean() for all_rank in all_ranks]).mean()
    return macro_mr

def micro_mr(all_ranks):
    micro_mr = np.array(list(itertools.chain(*all_ranks))).mean()
    return micro_mr

def hit_at_1(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 1)
    return 1.0 * hits / len(rank_positions)

def hit_at_3(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 3)
    return 1.0 * hits / len(rank_positions)

def hit_at_5(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 5)
    return 1.0 * hits / len(rank_positions)

def hit_at_10(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 10)
    return 1.0 * hits / len(rank_positions)

def precision_at_1(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 1)
    return 1.0 * hits / len(all_ranks)

def precision_at_3(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 3)
    return 1.0 * hits / (len(all_ranks)*3)

def precision_at_5(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 5)
    return 1.0 * hits / (len(all_ranks)*5)

def precision_at_10(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 10)
    return 1.0 * hits / (len(all_ranks)*10)

def mrr_scaled_10(all_ranks):
    """ Scaled MRR score, check eq. (2) in the PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf
    """
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions / 10)
    return (1.0 / scaled_rank_positions).mean()

def combined_metrics(all_ranks):
    """ 
    combination of three metrics, used in early stop 
    """
    score = macro_mr(all_ranks) * (1.0/max(mrr_scaled_10(all_ranks), 0.0001)) * (1.0/max(hit_at_3(all_ranks), 0.0001)) * (1.0/max(hit_at_1(all_ranks), 0.0001))
    return score