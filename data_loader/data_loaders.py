from torch.utils.data import DataLoader
from .dataset import *
import dgl
import torch
from itertools import chain 
from tqdm import tqdm
import numpy as np

BATCH_GRAPH_NODE_LIMIT = 100000

def save_pickle(data, path):
    """
    Save cleaned data

    Args:
        data: a dict of values (e.g. sentences, values)
        path: a file path to save
    """
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    """
    Load saved data

    Args:
        path: a file path to load data from. The data should be saved as one value each line format
    """
    with open(path, 'rb') as handle:
        return pickle.load(handle)

class UnifiedDataLoader(DataLoader):
    def __init__(self, mode, data_path, sampling_mode=1, batch_size=10, negative_size=20, max_pos_size=100,
                 expand_factor=50, shuffle=True, num_workers=8, cache_refresh_time=64, normalize_embed=False,
                 test_topk=-1, test=0):
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.batch_size = batch_size
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.shuffle = shuffle
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed
        test_flag = 'test' if test else 'train'
        
        # original: use binary file
        # raw_graph_dataset = MAGDataset(name="", path=data_path, raw=False)

        # raw mode without using binary file
        self.embed_suffix = data_path.split('/')[-1].split('.')[1]
        self.name = data_path.split('/')[-1].split('.')[0]
        raw_graph_dataset = MAGDataset(name=self.name, path=data_path, embed_suffix=self.embed_suffix, raw=True,
                                        existing_partition=False, partition_pattern='internal')
        
        if 'g' in mode and 'p' in mode:
            msk_graph_dataset = GraphPathDataset(raw_graph_dataset, mode=test_flag,sampling_mode=sampling_mode,
                                                 negative_size=negative_size, max_pos_size=max_pos_size,
                                                 expand_factor=expand_factor, cache_refresh_time=cache_refresh_time,
                                                 normalize_embed=normalize_embed, test_topk=test_topk)
        elif 'g' in mode:
            msk_graph_dataset = GraphDataset(raw_graph_dataset, mode=test_flag,sampling_mode=sampling_mode,
                                             negative_size=negative_size,
                                             max_pos_size=max_pos_size, expand_factor=expand_factor,
                                             cache_refresh_time=cache_refresh_time, normalize_embed=normalize_embed,
                                             test_topk=test_topk)
        elif 'p' in mode:
            msk_graph_dataset = PathDataset(raw_graph_dataset, mode=test_flag,sampling_mode=sampling_mode, negative_size=negative_size,
                                            max_pos_size=max_pos_size, expand_factor=expand_factor,
                                            cache_refresh_time=cache_refresh_time, normalize_embed=normalize_embed,
                                            test_topk=test_topk)
        else:
            msk_graph_dataset = RawDataset(raw_graph_dataset, mode=test_flag,sampling_mode=sampling_mode, negative_size=negative_size,
                                           max_pos_size=max_pos_size, expand_factor=expand_factor,
                                           cache_refresh_time=cache_refresh_time, normalize_embed=normalize_embed,
                                           test_topk=test_topk)
        self.dataset = msk_graph_dataset
        self.vocab = self.dataset.vocab # a list map from node_id to human-readable concept string
        print(f'size of vocab: {len(self.vocab)}') #13936
        self.num_workers = num_workers
        super(UnifiedDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                collate_fn=self.collate_fn, num_workers=self.num_workers,
                                                pin_memory=True)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader

        # add context sentences
        wordid2context = {}
        data_dir_cache = 'data/SemEval-Verb/context_wordid2sen_02261200_11453.pickle'
        if os.path.isfile(data_dir_cache):
            wordid2context = load_pickle(data_dir_cache)
            print('loaded context sentence wordid2context from ', data_dir_cache)

        data_dir_cache = 'data/SemEval-Verb/word2sen_data_loader_ct_temp.pickle'
        if os.path.isfile(data_dir_cache) and True:
            self.word2sen = load_pickle(data_dir_cache)
            print('word2sen Data loaded from cache', data_dir_cache)
        else:
            # go through each word in the vocab and save their sentences
            self.word2sen = {}
            not_found_key_words = []
            for i, word in enumerate(tqdm(self.vocab)):
                word_id = get_wn_id_from_str(word)

                # extract context sentence
                context_sens = []
                if word_id in wordid2context:
                    context_sens = wordid2context[word_id][:2]

                # print(word_id)
                sen, bf_lemma, contain_lemma, not_found_key_flag = assemble_sentence_single(word_id, context_sens)
                if not_found_key_flag:
                    not_found_key_words.append(word_id)
                # print(sen)
                self.word2sen[i] = [word, word_id, sen, bf_lemma, contain_lemma]
            save_pickle(self.word2sen, data_dir_cache)
        
            print(f'Number of word that not found key and related sentences: {len(not_found_key_words)} / {len(self.vocab)}')
            print(not_found_key_words[:min(50, len(not_found_key_words))])

        # sens statistics
        word2sen_len = [len(''.join(sen).split(' ')) for sen in self.word2sen.values()]
        print('Ave len of word2sen')
        print(np.mean(word2sen_len))

    def collate_fn(self, samples):
        if 'g' in self.mode and 'p' in self.mode:
            us, vs, graphs_u, graphs_v, paths_u, paths_v, lens, queries, labels = map(list, zip(*chain(*samples)))
            lens = torch.tensor(lens)
            max_u, max_v = lens.max(dim=0)[0]
            paths_u = [p + [self.dataset.pseudo_leaf_node] * (max_u - len(p)) for p in paths_u]
            paths_v = [p + [self.dataset.pseudo_leaf_node] * (max_v - len(p)) for p in paths_v]
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), \
                   graphs_u, graphs_v, torch.tensor(paths_u), torch.tensor(paths_v), lens
        elif 'g' in self.mode:
            us, vs, graphs_u, graphs_v, queries, labels = map(list, zip(*chain(*samples)))
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), graphs_u, graphs_v, \
                   None, None, None
        elif 'p' in self.mode:
            us, vs, paths_u, paths_v, lens, queries, labels = map(list, zip(*chain(*samples)))
            lens = torch.tensor(lens)
            max_u, max_v = lens.max(dim=0)[0]
            paths_u = [p + [self.dataset.pseudo_leaf_node] * (max_u - len(p)) for p in paths_u]
            paths_v = [p + [self.dataset.pseudo_leaf_node] * (max_v - len(p)) for p in paths_v]
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), None, None, \
                   torch.tensor(paths_u), torch.tensor(paths_v), lens
        else:
            us, vs, queries, labels = map(list, zip(*chain(*samples)))
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), None, None, None, None, None

    def __str__(self):
        return "\n\t".join([
            f"UnifiedDataLoader mode: {self.mode}",
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
            f"expand_factor: {self.expand_factor}",
            f"cache_refresh_time: {self.cache_refresh_time}",
            f"normalize_embed: {self.normalize_embed}",
        ])


class TaxoExpanDataLoader(DataLoader):
    def __init__(self, mode, data_path, sampling_mode=1, batch_size=10, negative_size=20, max_pos_size=100,
                 expand_factor=50, shuffle=True, num_workers=8, cache_refresh_time=64, normalize_embed=False,
                 test_topk=-1, split_in_oov=False, negative_parent=False):
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.batch_size = batch_size
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.shuffle = shuffle
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed
        self.negative_parent = negative_parent

        # original: use binary file
        # raw_graph_dataset = MAGDataset(name="", path=data_path, raw=False)

        # raw mode without using binary file
        self.embed_suffix = data_path.split('/')[-1].split('.')[1]
        self.name = data_path.split('/')[-1].split('.')[0]
        raw_graph_dataset = MAGDataset(name=self.name, path=data_path, embed_suffix=self.embed_suffix, raw=True,
                                        existing_partition=False)

        msk_graph_dataset = ExpanDataset(raw_graph_dataset, sampling_mode=sampling_mode,
                                             negative_size=negative_size,
                                             max_pos_size=max_pos_size, expand_factor=expand_factor,
                                             cache_refresh_time=cache_refresh_time, normalize_embed=normalize_embed,
                                             test_topk=test_topk, split_in_oov=split_in_oov, 
                                             negative_parent=negative_parent)
        self.dataset = msk_graph_dataset
        self.vocab = self.dataset.vocab # a list map from node_id to human-readable concept string
        print(f'size of vocab: {len(self.vocab)}') #13936

        # load definition sentences if needed
        if 'l' in self.mode:
            id_list, def_list = raw_graph_dataset.load_definitions()
            id_list = id_list + ['root', 'leaf']
            def_list = def_list + ['root', 'leaf']
            self.word2sen = {}
            for i, word in enumerate(tqdm(self.vocab)):
                word_id = get_wn_id_from_str(word)
                self.word2sen[i] = def_list[id_list.index(word_id)]

        self.num_workers = num_workers
        super(TaxoExpanDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                collate_fn=self.collate_fn, num_workers=self.num_workers,
                                                pin_memory=True)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader

    def collate_fn(self, samples):
        us, graphs_u, paths_u, lens, queries, labels = map(list, zip(*chain(*samples)))
        if 'g' not in self.mode:
            graphs_u = None
        if 'r' in self.mode:
            lens = torch.tensor(lens)
            max_u = lens.max(dim=0)[0]
            paths_u = [p + [self.dataset.pseudo_leaf_node] * (max_u - len(p)) for p in paths_u]
            paths_u = torch.tensor(paths_u)
        else:
            lens = None
            paths_u = None
        return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), graphs_u, paths_u, lens

    def __str__(self):
        return "\n\t".join([
            f"TaxoExpanDataLoader mode: {self.mode}",
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
            f"expand_factor: {self.expand_factor}",
            f"cache_refresh_time: {self.cache_refresh_time}",
            f"normalize_embed: {self.normalize_embed}",
        ])