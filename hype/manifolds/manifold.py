#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
import torch
import io
import os
from torch.nn import Embedding
import pickle

def get_verb_from_id(wordid, source='wn'):
    # get_word_from_class_name or wordnet or verbnet
    result = ''
    if '.' in wordid or '-' in wordid:
        if source == 'wn':
            result = wordid.split('.')[0]
            result = result.replace('-', ' ')
        elif source == 'vn':
            result = wordid.split('-')[0]
            result = result.replace('_', ' ')
    else:
        result = wordid
    return result

def get_wn_id_from_str(node_str):
    if node_str in ['root', 'leaf']:
        return node_str
    else:
        return node_str.split('||')[1].split('@@@')[0]
    # if it's mapping dataset
    # return node_str.split('@@@')[0]

def load_pretrained_matrix(voc_list, vector_dic, pretrained_matrix, emb_dim=50, word_id_form='id'):
    # pretrained_matrix = np.random.rand(len(voc_list), emb_dim)
    # word_id_form: word, id
    count_in_voc = 0
    for i, word in enumerate(voc_list):
        if word_id_form == 'id':
            # the pretrained vector dic keys are wordnet ids
            word = get_wn_id_from_str(word)
        else:
            # the pretrained vector dic keys are words lemma
            word = get_verb_from_id(get_wn_id_from_str(word))
        if word in vector_dic:
            pretrained_matrix[i] = vector_dic[word]
            pretrained_matrix[i] =  pretrained_matrix[i] * 1e-3
            count_in_voc += 1
        # else:
        #     print(word)
    print('Pre-trained vector coverage: ', count_in_voc, len(voc_list))
    return pretrained_matrix

def load_vectors(loaded_file_lst, skip_line=0):
    data = {}
    count = 0
    dim = 0
    for loaded_file in loaded_file_lst:
        print('loading word vectors from %s' % loaded_file)
        fin = io.open('%s' % (loaded_file), 'r', encoding='utf-8', newline='\n', errors='ignore')
        for line in fin:
            if count >= skip_line:
                tokens = line.rstrip().split(' ')
                tokens = list(filter(None, tokens))
                data[tokens[0]] = list(map(float, tokens[1:]))
            count += 1
            # Get word vectors dimension
            if count == 2:
                dim = len(list(map(float, tokens[1:])))
            if count % 10000 == 0: # to show the progress when loading the vector file
                print (count, end=" ")
    return data, dim


class Manifold(object):
    def allocate_lt(self, N, dim, sparse):
        return Embedding(N, dim, sparse=sparse)

    def normalize(self, u):
        return u

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    def init_weights(self, w, objects, scale=1e-4, evalonly=False, load_vec=False, vec_choice='poincareglove', use_cache=False):
        w.weight.data.uniform_(-scale, scale)

        if (not evalonly) and load_vec:
            # init weights using pre-trained embeddings
            w_np = w.weight.detach().numpy()

            if vec_choice == 'fasttext':
                emb_file_path = '/home/username/events/hierarchy/TMN/data/SemEval-Verb/wordnet_verb.terms.%s_mode4.embed' % vec_choice
                w_cache_path = 'cache/%s_w_np_03260100.pickle' % vec_choice
                word_id_form = 'id'

            if use_cache and os.path.exists(w_cache_path):
                # load weight from cache if the path exists
                with open(w_cache_path, 'rb') as handle:
                    w_np = pickle.load(handle)
                    print(f'loaded w_np pickle from {w_cache_path}')
            else:
                # load vectors from file and generate weights
                pretrained_emb_dict, pretrained_dim = load_vectors([emb_file_path],skip_line=1)
                print(f'pre-trained embedding dim: {pretrained_dim}')
                w_np = load_pretrained_matrix(objects, pretrained_emb_dict, w_np, 
                                            emb_dim=pretrained_dim, word_id_form=word_id_form)
                print('w_np generated')
                # save generated weight to cache
                with open(w_cache_path, 'wb') as handle:
                    pickle.dump(w_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f'w_np saved to {w_cache_path}')
            w.weight.data = torch.tensor(w_np)

    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport
        """
        raise NotImplementedError

    def norm(self, u, **kwargs):
        if isinstance(u, Embedding):
            u = u.weight
        return u.pow(2).sum(dim=-1).sqrt()

    @abstractmethod
    def half_aperture(self, u):
        """
        Compute the half aperture of an entailment cone.
        As in: https://arxiv.org/pdf/1804.01882.pdf
        """
        raise NotImplementedError

    @abstractmethod
    def angle_at_u(self, u, v):
        """
        Compute the angle between the two half lines (0u and uv
        """
        raise NotImplementedError
