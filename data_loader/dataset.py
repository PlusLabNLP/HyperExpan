import networkx as nx
from networkx.algorithms import descendants, ancestors
import dgl
from gensim.models import KeyedVectors
import numpy as np 
import torch 
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import pickle
import time
from tqdm import tqdm
import random
import copy
from itertools import chain, product, combinations
import os
import multiprocessing as mp
from functools import partial
from collections import defaultdict, deque
import more_itertools as mit
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()


MAX_TEST_SIZE = 1000
MAX_VALIDATION_SIZE = 1000


def add_edge_for_dgl(g, n1, n2):
    """
    https://github.com/dmlc/dgl/issues/1476 there is a bug in dgl add edges, so we need a wrapper
    """
    if not ((isinstance(n1, list) and len(n1) == 0) or (isinstance(n2, list) and len(n2) == 0)):
        g.add_edges(n1, n2)


def single_source_shortest_path_length(source,G,cutoff=None):
    """Compute the shortest path lengths from source to all reachable nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    lengths : dictionary
        Dictionary of shortest path lengths keyed by target.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> length=nx.single_source_shortest_path_length(G,0)
    >>> length[4]
    4
    >>> print(length)
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    See Also
    --------
    shortest_path_length
    """
    seen={}                  # level (number of hops) when seen in BFS
    level=0                  # the current level
    nextlevel={source:1}  # dict of nodes to check at next level
    while nextlevel:
        thislevel=nextlevel  # advance to next level
        nextlevel={}         # and start a new list (fringe)
        for v in thislevel:
            if v not in seen:
                seen[v]=level # set the level of vertex v
                nextlevel.update(G[v]) # add neighbors of v
        if (cutoff is not None and cutoff <= level):  break
        level=level+1
    return (source, seen)  # return all path lengths as dictionary


def parallel_all_pairs_shortest_path_length(g, node_ids, num_workers=20):
    # TODO This can be trivially parallelized.
    res = {}
    pool = mp.Pool(processes=num_workers)
    p = partial(single_source_shortest_path_length, G=g, cutoff=None)
    result_list = pool.map(p, node_ids)
    for result in result_list:
        res[result[0]] = result[1]
    pool.close()
    pool.join()
    return res


def save_word_list(word_list, path='data/verbnet_uniq_verbs.pickle'):
    with open(path, 'wb') as handle:
        pickle.dump(word_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_word_list_txt(word_list, path='data/verbnet_uniq_verbs.txt'):
    with open(path, "w") as f: 
        f.write('\n'.join(word_list))

def save_word_list_both(word_list, path='data/verbnet_uniq_verbs'):
    save_word_list(word_list, '%s.pickle' % path)
    save_word_list_txt(word_list, '%s.txt' % path)

def get_verb_from_id(wordid, source='wn', remove_under=True):
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
            if remove_under:
                result = result.replace('_', ' ')
        elif source == 'vn':
            result = wordid.split('-')[0]
            result = result.replace('_', ' ')
    else:
        result = wordid
    return result

def get_level(graph: nx.DiGraph, node, show=False):
    # print(node)
    if node.level > 0:
        return node.level
    if len(list(graph.predecessors(node))) > 0:
        current_level = list()
        for parent in graph.predecessors(node):
            current_level.append(get_level(graph, parent, show=show) + 1)
        node.level = min([l for l in current_level if l >= 0])
        if len(current_level) > 1:
            if show: print(
                f"Current node: {node}, potential levels: {', '.join([str(l) for l in current_level])}, take minimum.")
        if show: print(node)
        return node.level
    else:
        node.level = 0
        return node.level

def get_ancestors(graph: nx.DiGraph, node, show=False):
    """
    Return: a list of all ancestors of node on graph
    """
    # print(node)
    if len(list(graph.predecessors(node))) > 0:
        current_ancestors = list()
        for parent in graph.predecessors(node):
            current_ancestors.append(get_ancestors(graph, parent, show=show) + [node])
        ancestors = max(current_ancestors, key=len)
        if len(current_ancestors) > 1:
            if show: print(
                f"Current node: {node}, potential levels: {', '.join([str(len(l)) for l in current_ancestors])}, take maximum.")
        if show: print(node)
        return ancestors
    else:
        ancestors = [node]
        return ancestors

class Taxon(object):
    def __init__(self, tx_id, rank=-1, norm_name="none", display_name="None", main_type="", level="-100", p_count=0, c_count=0, create_date="None"):
        self.tx_id = tx_id
        self.rank = int(rank)
        self.norm_name = norm_name
        self.display_name = display_name
        self.main_type = main_type
        self.level = int(level)
        self.p_count = int(p_count)
        self.c_count = int(c_count)
        self.create_date = create_date
        
    def __str__(self):
        return "Taxon {} (name: {}, level: {})".format(self.tx_id, self.norm_name, self.level)
        
    def __lt__(self, another_taxon):
        if self.level < another_taxon.level:
            return True
        else:
            return self.rank < another_taxon.rank


class MAGDataset(object):
    def __init__(self, name, path, embed_suffix="", raw=True, existing_partition=False, partition_pattern='leaf', shortest_path=False):
        """ Raw dataset class for MAG dataset

        Parameters
        ----------
        name : str
            taxonomy name
        path : str
            path to dataset, if raw=True, this is the directory path to dataset, if raw=False, this is the pickle path
        embed_suffix : str
            suffix of embedding file name, by default ""
        raw : bool, optional
            load raw dataset from txt (True) files or load pickled dataset (False), by default True
        existing_partition : bool, optional
            whether to use the existing the train/validation/test partitions or randomly sample new ones, by default False
        """
        self.name = name  # taxonomy name
        self.path = path
        self.path_folder = '/'.join(path.split('/')[:-1])
        self.embed_suffix = embed_suffix
        self.existing_partition = existing_partition
        self.partition_pattern = partition_pattern
        self.g_full = dgl.DGLGraph()  # full graph, including masked train/validation node indices
        self.vocab = []  # from node_id to human-readable concept string
        self.train_node_ids = []  # a list of train node_ids
        self.validation_node_ids = []  # a list of validation node_ids
        self.test_node_ids = []  # a list of test node_ids
        self.shortest_path = shortest_path
        self.candidates_separate_available = False
        # add a suffix in bin file name if it's leaf partition_pattern
        self.partition_pattern_suffix = ""
        if self.partition_pattern == 'leaf':
            self.partition_pattern_suffix = '.ex'

        if raw:
            if '.bin' in self.path:
                # path is binary file path, not a directory
                self._load_dataset_raw(self.path_folder)
            else:
                # paht is a directory, used when generating binary file
                self._load_dataset_raw(path)
        else:
            self._load_dataset_pickled(path)

    def _load_dataset_pickled(self, pickle_path):
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)

        self.name = data["name"]
        self.g_full = data["g_full"]
        self.vocab = data["vocab"]
        self.train_node_ids = data["train_node_ids"]
        self.validation_node_ids = data["validation_node_ids"]
        self.test_node_ids = data["test_node_ids"]
        if "candidates_node_ids" in data:
            self.candidates_node_ids = data["candidates_node_ids"]
        if self.shortest_path:
            self.shortest_path = data['shortest_path']

    def _load_dataset_raw(self, dir_path):
        """ Load data from three seperated files, generate train/validation/test partitions, and save to binary pickled dataset.
        Please refer to the README.md file for details.


        Parameters
        ----------
        dir_path : str
            The path to a directory containing three input files.
        """
        node_file_name = os.path.join(dir_path, f"{self.name}.terms")
        edge_file_name = os.path.join(dir_path, f"{self.name}.taxo")
        if self.embed_suffix == "":
            embedding_file_name = os.path.join(dir_path, f"{self.name}.terms.embed")
            output_pickle_file_name = os.path.join(dir_path, f"{self.name}{self.partition_pattern_suffix}.pickle.bin")
        else:
            embedding_file_name = os.path.join(dir_path, f"{self.name}.terms.{self.embed_suffix}.embed")
            output_pickle_file_name = os.path.join(dir_path, f"{self.name}.{self.embed_suffix}{self.partition_pattern_suffix}.pickle.bin")
        if self.existing_partition:
            train_node_file_name = os.path.join(dir_path, f"{self.name}.terms.train")
            validation_node_file_name = os.path.join(dir_path, f"{self.name}.terms.validation")
            test_file_name = os.path.join(dir_path, f"{self.name}.terms.test")
            candidates_file_name = os.path.join(dir_path, f"{self.name}.terms.candidates")

        tx_id2taxon = {}
        taxonomy = nx.DiGraph()

        # load nodes
        with open(node_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading terms"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    taxon = Taxon(tx_id=segs[0], norm_name=segs[1], display_name=segs[1])
                    tx_id2taxon[segs[0]] = taxon
                    taxonomy.add_node(taxon)

        # load edges
        with open(edge_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading relations"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = tx_id2taxon[segs[0]]
                    child_taxon = tx_id2taxon[segs[1]]
                    taxonomy.add_edge(parent_taxon, child_taxon)

        self.taxonomy = taxonomy
        self.tx_id2taxon = tx_id2taxon

        # get the level of nodes
        for node in tqdm(taxonomy.nodes, desc="Calculating node level"):
            # print(node)
            node.level = get_level(taxonomy, node)
            # print(node.level)

        # get root and max level
        for node in taxonomy.nodes:
            if node.level == 0:
                self.root = node
        self.max_level = max([node.level for node in taxonomy.nodes])
        print(f"max_level: {self.max_level}")

        # load embedding features
        print("Loading embedding ...")
        embeddings = KeyedVectors.load_word2vec_format(embedding_file_name)
        print(f"Finish loading embedding of size {embeddings.vectors.shape}")

        # load train/validation/test partition files if needed
        if self.existing_partition:
            print("Loading existing train/validation/test partitions")
            raw_train_node_list = self._load_node_list(train_node_file_name)
            raw_validation_node_list = self._load_node_list(validation_node_file_name)
            raw_test_node_list = self._load_node_list(test_file_name)
            # identify candidate list if the file exists
            if os.path.exists(candidates_file_name):
                self.candidates_separate_available = True
                raw_candidates_node_list = self._load_node_list(candidates_file_name)

        # generate vocab, tx_id is the old taxon_id read from {self.name}.terms file, node_id is the new taxon_id from 0 to len(vocab)
        tx_id2node_id = {node.tx_id:idx for idx, node in enumerate(taxonomy.nodes()) }
        self.tx_id2node_id = tx_id2node_id
        node_id2tx_id = {v:k for k, v in tx_id2node_id.items()}
        self.node_id2tx_id = node_id2tx_id
        self.vocab = [tx_id2taxon[node_id2tx_id[node_id]].norm_name + "@@@" + str(node_id) for node_id in node_id2tx_id]
        self.vocab_level = [tx_id2taxon[node_id2tx_id[node_id]].level for node_id in node_id2tx_id]

        # generate dgl.DGLGraph()
        edges = []
        for edge in taxonomy.edges():
            parent_node_id = tx_id2node_id[edge[0].tx_id]
            child_node_id = tx_id2node_id[edge[1].tx_id]
            edges.append([parent_node_id, child_node_id])

        node_features = np.zeros(embeddings.vectors.shape)
        for node_id, tx_id in node_id2tx_id.items():
            node_features[node_id, :] = embeddings[tx_id]
        node_features = torch.FloatTensor(node_features)

        self.g_full.add_nodes(len(node_id2tx_id), {'x': node_features})
        self.g_full.add_edges([e[0] for e in edges], [e[1] for e in edges])

        # generate validation/test node_indices using either existing partitions or randomly sampled partition
        if self.existing_partition:
            self.train_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_train_node_list]
            self.validation_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_validation_node_list]
            self.test_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_test_node_list]
            if self.candidates_separate_available:
                self.candidates_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_candidates_node_list]
        else:
            print("Partition graph ...")
            if self.partition_pattern == 'leaf':
                leaf_node_ids = []
                for node in taxonomy.nodes():
                    if taxonomy.out_degree(node) == 0:
                        leaf_node_ids.append(tx_id2node_id[node.tx_id])

                # # save nodes if want to save output of the binary generation process to use in other model
                # leaf_nodes = [node_id2tx_id[node_id] for node_id in leaf_node_ids]
                # print(len(leaf_nodes))
                # print(leaf_nodes[0:20])
                # save_word_list_both(leaf_nodes, 'leaf_nodes')

                random.seed(47) # original default
                # random.seed(26)
                random.shuffle(leaf_node_ids)
                validation_size = min(int(len(leaf_node_ids) * 0.1), MAX_VALIDATION_SIZE)
                test_size = min(int(len(leaf_node_ids) * 0.1), MAX_TEST_SIZE)
                self.validation_node_ids = leaf_node_ids[:validation_size]
                self.test_node_ids = leaf_node_ids[validation_size:(validation_size+test_size)]
                self.train_node_ids = [node_id for node_id in node_id2tx_id if node_id not in self.validation_node_ids and node_id not in self.test_node_ids]
                
                # # save nodes if want to save output of the binary generation process to use in other model
                # validation_nodes = [node_id2tx_id[node_id] for node_id in self.validation_node_ids]
                # print(len(validation_nodes))
                # print(validation_nodes[0:20])
                # save_word_list_both(validation_nodes, 'validation_nodes')

                # test_nodes = [node_id2tx_id[node_id] for node_id in self.test_node_ids]
                # print(len(test_nodes))
                # print(test_nodes[0:20])
                # save_word_list_both(test_nodes, 'test_nodes')

                # train_nodes = [node_id2tx_id[node_id] for node_id in self.train_node_ids]
                # print(len(train_nodes))
                # print(train_nodes[0:20])
                # save_word_list_both(train_nodes, 'train_nodes')
            elif self.partition_pattern == 'internal':
                root_node = [node for node in taxonomy.nodes() if taxonomy.in_degree(node) == 0]
                sampled_node_ids = [tx_id2node_id[node.tx_id] for node in taxonomy.nodes() if node not in root_node]
                random.seed(47)
                random.shuffle(sampled_node_ids)

                validation_size = min(int(len(sampled_node_ids) * 0.1), MAX_VALIDATION_SIZE)
                test_size = min(int(len(sampled_node_ids) * 0.1), MAX_TEST_SIZE)
                self.validation_node_ids = sampled_node_ids[:validation_size]
                self.test_node_ids = sampled_node_ids[validation_size:(validation_size+test_size)]
                self.train_node_ids = [node_id for node_id in node_id2tx_id if node_id not in self.validation_node_ids and node_id not in self.test_node_ids]
            else:
                raise ValueError('Unknown partition method!')
            print("Finish partition graph ...")

        # Compute shortest path distances
        if self.shortest_path:
            dag = self._get_holdout_subgraph(self.train_node_ids).to_undirected()
            numnodes = len(node_id2tx_id)
            spdists = -1 * (np.ones((numnodes, numnodes), dtype=np.float))
            res = parallel_all_pairs_shortest_path_length(dag, self.train_node_ids)
            for u, dists in res.items():
                for v, dist in dists.items():
                    spdists[u][v] = int(dist)

            spdists[spdists == -1] = int(spdists.max())
            self.shortest_path = spdists

        # save to pickle for faster loading next time
        print("start saving pickle data")
        with open(output_pickle_file_name, 'wb') as fout:
            # Pickle the 'data' dictionary using the highest protocol available.
            data = {
                "name": self.name,
                "g_full": self.g_full,
                "vocab": self.vocab,
                "train_node_ids": self.train_node_ids,
                "validation_node_ids": self.validation_node_ids,
                "test_node_ids": self.test_node_ids,
                "shortest_path": self.shortest_path
            }
            if self.candidates_separate_available:
                data["candidates_node_ids"] = self.candidates_node_ids
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    def load_definitions(self):
        definitions_file_name = os.path.join(self.path_folder, f"{self.name}.definitions")
        def_list = self._load_node_list(definitions_file_name, mode='def')
        id_list = self._load_node_list(definitions_file_name, mode='id')
        return id_list, def_list

    def _load_node_list(self, file_path, mode=None):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if mode == 'id':
                    # deal with separate training data that has \t as separate token
                    segs = line.split("\t")
                    if line:
                        # node_list.append(line)
                        node_list.append(segs[0])
                elif mode == 'def':
                    segs = line.split("\t")
                    if line:
                        # node_list.append(line)
                        node_list.append(segs[1])
                else:
                    # original implementation
                    if line:
                        node_list.append(line)
        return node_list

    def _get_holdout_subgraph(self, node_ids):
        full_graph = self.g_full.to_networkx()
        node_to_remove = [n for n in full_graph.nodes if n not in node_ids]
        subgraph = full_graph.subgraph(node_ids).copy()
        for node in node_to_remove:
            parents = set()
            children = set()
            ps = deque(full_graph.predecessors(node))
            cs = deque(full_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(full_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(full_graph.successors(c))
            for p, c in product(parents, children):
                subgraph.add_edge(p, c)
        # remove jump edges
        node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
        for node in subgraph.nodes():
            if subgraph.out_degree(node) > 1:
                successors1 = set(subgraph.successors(node))
                successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
                checkset = successors1.intersection(successors2)
                if checkset:
                    for s in checkset:
                        subgraph.remove_edge(node, s)
        return subgraph


class RawDataset(Dataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        start = time.time()
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed
        self.test_topk = test_topk

        self.tx_id2node_id = graph_dataset.tx_id2node_id
        self.node_id2tx_id = graph_dataset.node_id2tx_id
        self.tx_id2taxon = graph_dataset.tx_id2taxon
        self.taxonomy = graph_dataset.taxonomy

        self.node_features = graph_dataset.g_full.ndata['x']
        full_graph = graph_dataset.g_full.to_networkx()
        train_node_ids = graph_dataset.train_node_ids
        roots = [node for node in full_graph.nodes() if full_graph.in_degree(node) == 0]
        if len(roots) > 1:
            self.root = len(full_graph.nodes)
            for r in roots:
                full_graph.add_edge(self.root, r)
            root_vector = torch.mean(self.node_features[roots], dim=0, keepdim=True)
            self.node_features = torch.cat((self.node_features, root_vector), 0)
            self.vocab = graph_dataset.vocab + ['root', 'leaf']
            train_node_ids.append(self.root)
        else:
            self.root = roots[0]
            self.vocab = graph_dataset.vocab + ['leaf']
        self.full_graph = full_graph

        # get some useful arguments from graph_dataset
        self.vocab_level = graph_dataset.vocab_level
        self.max_level = graph_dataset.max_level
        print("vocab_level count")
        values, counts = np.unique(self.vocab_level, return_counts=True)
        print(values)
        print(counts)

        if mode == 'train':
            # add pseudo leaf node to core graph
            self.core_subgraph = self._get_holdout_subgraph(train_node_ids)
            self.pseudo_leaf_node = len(full_graph.nodes)
            for node in list(self.core_subgraph.nodes()):
                self.core_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.leaf_nodes = [node for node in self.core_subgraph.nodes() if self.core_subgraph.out_degree(node) == 1]
            # for pseudo leaf node
            leaf_vector = torch.zeros((1, self.node_features.size(1))) # zero vector works best
            self.node_features = torch.cat((self.node_features, leaf_vector), 0)
            if self.normalize_embed:
                self.node_features = F.normalize(self.node_features, p=2, dim=1)

            # add interested node list and subgraph
            # remove supersource nodes (i.e., nodes without in-degree 0)
            interested_node_set = set(train_node_ids) - set([self.root])
            self.node_list = list(interested_node_set)

            # build node2pos, node2nbs, node2edge
            self.node2pos, self.node2edge = {}, {}
            self.node2parents, self.node2children, self.node2nbs = {}, {}, {self.pseudo_leaf_node:[]}
            for node in interested_node_set:
                parents = set(self.core_subgraph.predecessors(node))
                children = set(self.core_subgraph.successors(node))
                if len(children) > 1:
                    children = [i for i in children if i != self.pseudo_leaf_node]
                node_pos_edges = [(pre, suc) for pre in parents for suc in children if pre!=suc]
                self.node2edge[node] = set(self.core_subgraph.in_edges(node)).union(set(self.core_subgraph.out_edges(node)))
                self.node2pos[node] = node_pos_edges
                self.node2parents[node] = parents
                self.node2children[node] = children
                self.node2nbs[node] = parents.union(children)
            self.node2nbs[self.root] = set([n for n in self.core_subgraph.successors(self.root) if n != self.pseudo_leaf_node])

            self.valid_node_list = graph_dataset.validation_node_ids
            holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.validation_node_ids)
            self.valid_node2pos = self._find_insert_posistion(graph_dataset.validation_node_ids, holdout_subgraph)

            self.test_node_list = graph_dataset.test_node_ids
            holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids)
            self.test_node2pos = self._find_insert_posistion(graph_dataset.test_node_ids, holdout_subgraph)

            # used for sampling negative positions during train/validation stage
            self.pointer = 0
            self.all_edges = list(self._get_candidate_positions(self.core_subgraph))
            self.edge2dist = {(u, v): nx.shortest_path_length(self.core_subgraph, u, v) for (u, v) in self.all_edges}
            random.shuffle(self.all_edges)
        elif mode == 'test':
            # add pseudo leaf node to core graph
            self.core_subgraph = self.full_graph
            self.pseudo_leaf_node = len(full_graph.nodes)
            self.node_list = list(self.core_subgraph.nodes())
            for node in self.node_list:
                self.core_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.leaf_nodes = [node for node in self.core_subgraph.nodes() if
                               self.core_subgraph.out_degree(node) == 1]
            # for pseudo leaf node
            leaf_vector = torch.zeros((1, self.node_features.size(1)))  # zero vector works best
            self.node_features = torch.cat((self.node_features, leaf_vector), 0)
            if self.normalize_embed:
                self.node_features = F.normalize(self.node_features, p=2, dim=1)

            # used for sampling negative positions during train/validation stage
            self.all_edges = list(self._get_candidate_positions(self.core_subgraph))

        print('all_edges loaded')
        print(len(self.all_edges))
        print(self.all_edges[:10])

        end = time.time()
        print(f"Finish loading dataset ({end - start} seconds)")

    def __str__(self):
        return f"{self.__class__.__name__} mode:{self.mode}"

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        One data instance is a list of (anchor_egonet, query_node_feature, label) triplets.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                res.append([u, v, query_node, (1, 1, 1, 1)])
        elif self.sampling_mode > 0:
            u, v = random.choice(self.node2pos[query_node])
            res.append([u, v, query_node, (1, 1, 1, 1)])

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u, v in negative_anchors:
            u_flag = int(u in self.node2parents[query_node])
            v_flag = int(v in self.node2children[query_node])
            e_flag = int(self.edge2dist[(u, v)] <= 2)
            res.append([u, v, query_node, (0, u_flag, v_flag, e_flag)])

        return tuple(res)

    def _get_holdout_subgraph(self, node_ids):
        node_to_remove = [n for n in self.full_graph.nodes if n not in node_ids]
        subgraph = self.full_graph.subgraph(node_ids).copy()
        for node in node_to_remove:
            parents = set()
            children = set()
            ps = deque(self.full_graph.predecessors(node))
            cs = deque(self.full_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(self.full_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(self.full_graph.successors(c))
            for p in parents:
                for c in children:
                    subgraph.add_edge(p, c)
        # remove jump edges
        node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
        for node in subgraph.nodes():
            if subgraph.out_degree(node) > 1:
                successors1 = set(subgraph.successors(node))
                successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
                checkset = successors1.intersection(successors2)
                if checkset:
                    for s in checkset:
                        subgraph.remove_edge(node, s)
        return subgraph

    def _get_candidate_positions(self, graph):
        node2descendants = {n: set(descendants(graph, n)) for n in graph.nodes}
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        return candidates

    def _find_insert_posistion(self, node_ids, holdout_graph, ignore=[]):
        node2pos = {}
        subgraph = self.core_subgraph
        for node in node_ids:
            if node in ignore:
                continue
            parents = set()
            children = set()
            ps = deque(holdout_graph.predecessors(node))
            cs = deque(holdout_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(holdout_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(holdout_graph.successors(c))
            if not children:
                children.add(self.pseudo_leaf_node)
            position = [(p, c) for p in parents for c in children if p!=c]
            node2pos[node] = position
        return node2pos

    def _get_negative_anchors(self, query_node, negative_size):
        if self.sampling_mode == 0:
            return self._get_at_most_k_negatives(query_node, negative_size)
        elif self.sampling_mode == 1:
            return self._get_exactly_k_negatives(query_node, negative_size)

    def _get_at_most_k_negatives(self, query_node, negative_size):
        """ Generate AT MOST negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        while True:
            negatives = [ele for ele in self.all_edges[self.pointer: self.pointer + negative_size] if
                         ele not in self.node2pos[query_node] and ele not in self.node2edge[query_node]]
            if len(negatives) > 0:
                break
            self.pointer += negative_size
            if self.pointer >= len(self.all_edges):
                self.pointer = 0

        return negatives

    def _get_exactly_k_negatives(self, query_node, negative_size, ignore=[]):
        """ Generate EXACTLY negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        negatives = []
        while len(negatives) != negative_size:
            n_lack = negative_size - len(negatives)
            negatives.extend([ele for ele in self.all_edges[self.pointer: self.pointer + n_lack] if
                                  ele not in self.node2pos[query_node] and ele not in self.node2edge[query_node] and ele not in ignore])
            self.pointer += n_lack
            if self.pointer >= len(self.all_edges):
                self.pointer = 0
                random.shuffle(self.all_edges)
        if len(negatives) > negative_size:
            negatives = negatives[:negative_size]

        return negatives


class GraphDataset(RawDataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        super(GraphDataset, self).__init__(graph_dataset, mode, sampling_mode, negative_size, max_pos_size,
                                           expand_factor, cache_refresh_time, normalize_embed, test_topk)

        # used for caching local subgraphs
        self.cache = {}  # if g = self.cache[anchor_node], then g is the egonet centered on the anchor_node
        self.cache_counter = {}  # if n = self.cache[anchor_node], then n is the number of times you used this cache

        lg = dgl.DGLGraph()
        # format for pesudo_leaf_node should be the same as following _get_subgraph function
        lg.add_nodes(1, {"_id": torch.tensor([self.pseudo_leaf_node]), 
                        "pos": torch.tensor([1])})
        # lg.add_nodes(1, {"_id": torch.tensor([self.pseudo_leaf_node]), 
        #                 "pos": torch.tensor([1]),
        #                 "dist": torch.tensor([0]),
        #                 "abs_level": torch.tensor([13])})
        lg.add_edges(lg.nodes(), lg.nodes(), {'erel': torch.tensor([2] * len(lg.nodes()))})
        self.cache[self.pseudo_leaf_node] = lg

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        One data instance is a list of (anchor_egonet, query_node_feature, label) triplets.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
                res.append([u, v, u_egonet, v_egonet, query_node, (1, 1, 1, 1)])
        elif self.sampling_mode > 0:
            u, v = random.choice(self.node2pos[query_node])
            u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
            res.append([u, v, u_egonet, v_egonet, query_node, (1, 1, 1, 1)])

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u, v in negative_anchors:
            u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
            u_flag = int(u in self.node2parents[query_node])
            v_flag = int(v in self.node2children[query_node])
            e_flag = int(self.edge2dist[(u, v)] <= 2)
            res.append([u, v, u_egonet, v_egonet, query_node, (0, u_flag, v_flag, e_flag)])

        return tuple(res)

    def _check_cache_flag(self, node):
        return (node in self.cache) and (self.cache_counter[node] < self.cache_refresh_time)

    def _get_subgraph_and_node_pair(self, query_node, anchor_node_u, anchor_node_v):
        """ Generate anchor_egonet and obtain query_node feature

        instance_mode: 0 means negative example, 1 means positive example
        """

        # [IMPORTANT]
        # if anchor_node_u == self.pseudo_leaf_node:
        #     return self.cache[anchor_node_u]

        if anchor_node_u == self.pseudo_leaf_node:
            g_u = self.cache[anchor_node_u]
        else:
            u_cache_flag = self._check_cache_flag(anchor_node_u)
            u_flag = ((query_node < 0) or (anchor_node_u not in self.node2nbs[query_node])) and (anchor_node_u not in self.node2nbs[anchor_node_v])
            if u_flag and u_cache_flag:
                g_u = self.cache[anchor_node_u]
                self.cache_counter[anchor_node_u] += 1
            else:
                g_u = self._get_subgraph(query_node, anchor_node_u, anchor_node_v, u_flag)
                if u_flag:  # save to cache
                    self.cache[anchor_node_u] = g_u
                    self.cache_counter[anchor_node_u] = 0

        if anchor_node_v == self.pseudo_leaf_node:
            g_v = self.cache[anchor_node_v]
        else:
            v_cache_flag = self._check_cache_flag(anchor_node_v)
            v_flag = ((query_node < 0) or (anchor_node_v not in self.node2nbs[query_node])) and (anchor_node_v not in self.node2nbs[anchor_node_u])
            if v_flag and v_cache_flag:
                g_v = self.cache[anchor_node_v]
                self.cache_counter[anchor_node_v] += 1
            else:
                g_v = self._get_subgraph(query_node, anchor_node_v, anchor_node_u, v_flag)
                if v_flag:  # save to cache
                    self.cache[anchor_node_v] = g_v
                    self.cache_counter[anchor_node_v] = 0

        return g_u, g_v

    def _get_subgraph_new(self, query_node, anchor_node, other_anchor_node, instance_mode):
        one_hop = False
        hop_number = 3
        if instance_mode:
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor)]
                # nodes_pos = []
                # nodes_dist = []
                # for n_hop in range(hop_number):
                #     nodes_pos.extend([n_hop + 2] * len(nodes))
                #     nodes_dist.extend([n_hop + 1] * len(nodes))
                nodes_pos = [0] * len(nodes)
                nodes_dist = [1] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                nodes_dist.append(0)

            else:
                # 1. get direct parent
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node)]
                nodes_pos = [0] * len(nodes)
                nodes_dist = [1] * len(nodes)
                if not one_hop:
                    # 6. children of direct parent
                    im = nodes
                    for n_hop in range(1, hop_number):
                        im2 = []
                        for node_sibling in im:
                            if self.core_subgraph.out_degree(node_sibling) <= self.expand_factor:
                                siblings_2 = [n for n in self.core_subgraph.successors(node_sibling) if
                                              n != self.pseudo_leaf_node]
                            else:
                                siblings_2 = [n for n in
                                              random.choices(list(self.core_subgraph.successors(node_sibling)),
                                                             k=self.expand_factor)
                                              if
                                              n != self.pseudo_leaf_node]
                            nodes.extend(siblings_2)
                            im2.extend(siblings_2)
                            nodes_pos.extend([n_hop + 2] * len(siblings_2))
                            nodes_dist.extend([n_hop + 1] * len(siblings_2))
                        im = im2
                    # 2. get ancestors, new!
                    # do not include anchor node and its direct parent in ancestors list
                    # do not include root node (first item)
                    ancestors = get_ancestors(self.core_subgraph, anchor_node)[1:-2]
                    ancestors.reverse()  # closest ancestors first
                    nodes.extend(ancestors)
                    nodes_pos.extend([hop_number + 2] * len(ancestors))
                    nodes_dist.extend(list(range(hop_number + 2, hop_number + 2 + len(ancestors))))

                # 3. anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                nodes_dist.append(0)
                # 4. siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if n != self.pseudo_leaf_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)),
                                               k=self.expand_factor)
                                if
                                n != self.pseudo_leaf_node]
                nodes.extend(siblings)
                nodes_pos.extend([hop_number + 3] * len(siblings))
                nodes_dist.extend([1] * len(siblings))
                im = siblings
                for n_hop in range(1, hop_number):
                    # 5. children of siblings
                    im2 = []
                    for node_sibling in im:
                        if self.core_subgraph.out_degree(node_sibling) <= self.expand_factor:
                            siblings_2 = [n for n in self.core_subgraph.successors(node_sibling) if
                                          n != self.pseudo_leaf_node]
                        else:
                            siblings_2 = [n for n in
                                          random.choices(list(self.core_subgraph.successors(node_sibling)),
                                                         k=self.expand_factor)
                                          if
                                          n != self.pseudo_leaf_node]
                        nodes.extend(siblings_2)
                        im2.extend(siblings_2)
                        nodes_pos.extend([n_hop + hop_number + 3] * len(siblings_2))
                        nodes_dist.extend([n_hop + 1] * len(siblings_2))
                    im = im2


        else:  # remove query_node from the children set of anchor_node
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor) if n != query_node and n!=other_anchor_node]
                nodes_pos = [0] * len(nodes)
                nodes_dist = [1] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                nodes_dist.append(0)
                # parent_node_idx = 0
                # nodes = [anchor_node]
                # nodes_pos = [1]
            # parents of anchor node
            else:
                # 1. get direct parent
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node) if n != query_node and n!=other_anchor_node]
                nodes_pos = [0] * len(nodes)
                nodes_dist = [1] * len(nodes)
                im = nodes
                for n_hop in range(1, hop_number):
                    # 6. children of direct parent
                    im2 = []
                    for node_sibling in im:
                        if self.core_subgraph.out_degree(node_sibling) <= self.expand_factor:
                            siblings_2 = [n for n in self.core_subgraph.successors(node_sibling) if
                                          n != self.pseudo_leaf_node]
                        else:
                            siblings_2 = [n for n in
                                          random.choices(list(self.core_subgraph.successors(node_sibling)),
                                                         k=self.expand_factor)
                                          if
                                          n != self.pseudo_leaf_node]
                        nodes.extend(siblings_2)
                        im2.extend(siblings_2)
                        nodes_pos.extend([n_hop + 2] * len(siblings_2))
                        nodes_dist.extend([n_hop + 1] * len(siblings_2))
                    im = im2
                    # 2. get ancestors, new!
                    # do not include anchor node and its direct parent in ancestors list
                    # do not include root node (first item)
                    ancestors = get_ancestors(self.core_subgraph, anchor_node)[1:-2]
                    ancestors.reverse()  # closest ancestors first
                    # print([n.display_name for n in ancestors])
                    nodes.extend(ancestors)
                    nodes_pos.extend([hop_number + 2] * len(ancestors))
                    nodes_dist.extend(list(range(hop_number + 2, hop_number + 2 + len(ancestors))))
                # 3. anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                nodes_dist.append(0)
                # 4. siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if
                                n != self.pseudo_leaf_node and n != query_node and n!=other_anchor_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)), k=self.expand_factor) if
                                n != self.pseudo_leaf_node and n != query_node and n!=other_anchor_node]
                nodes.extend(siblings)
                nodes_pos.extend([hop_number + 3] * len(siblings))
                nodes_dist.extend([1] * len(siblings))
                im = siblings
                for n_hop in range(1, hop_number):
                    # 5. children of siblings
                    im2 = []
                    for node_sibling in im:
                        if self.core_subgraph.out_degree(node_sibling) <= self.expand_factor:
                            siblings_2 = [n for n in self.core_subgraph.successors(node_sibling) if
                                          n != self.pseudo_leaf_node]
                        else:
                            siblings_2 = [n for n in
                                          random.choices(list(self.core_subgraph.successors(node_sibling)),
                                                         k=self.expand_factor)
                                          if
                                          n != self.pseudo_leaf_node]
                        nodes.extend(siblings_2)
                        im2.extend(siblings_2)
                        nodes_pos.extend([n_hop + hop_number + 3] * len(siblings_2))
                        nodes_dist.extend([n_hop + 1] * len(siblings_2))
                    im = im2

        nodes_level = [self.tx_id2taxon[self.node_id2tx_id[n]].level + 1 if self.vocab[n] != 'root' else 0 for n in
                       nodes]
        # if it's root node, level is 0
        # for other nodes, add 1 to level

        # create dgl graph with features
        g = dgl.DGLGraph()
        g.add_nodes(len(nodes), {"_id": torch.tensor(nodes),
                                 "pos": torch.tensor(nodes_pos),
                                 "dist": torch.tensor(nodes_dist),
                                 "abs_level": torch.tensor(nodes_level)})
        
        # add edge types
        length = len(list(range(parent_node_idx)))
        g.add_edges(list(range(parent_node_idx)), parent_node_idx, {'erel': torch.tensor([0] * length)})

        length = len(list(range(parent_node_idx + 1, len(nodes))))
        g.add_edges(parent_node_idx, list(range(parent_node_idx + 1, len(nodes))), {'erel': torch.tensor([1] * length)})

        # add self-cycle
        g.add_edges(g.nodes(), g.nodes(), {'erel': torch.tensor([2] * len(g.nodes()))})

        return g

    def _get_subgraph(self, query_node, anchor_node, other_anchor_node, instance_mode):
        if instance_mode:  # do not need to worry about query_node appears to be the child of anchor_node
            # parents of anchor node
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor)]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)

            else:
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node)]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                # siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if n != self.pseudo_leaf_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)), k=self.expand_factor) if
                                n != self.pseudo_leaf_node]
                nodes.extend(siblings)
                nodes_pos.extend([2] * len(siblings))
        else:  # remove query_node from the children set of anchor_node
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor) if n!=query_node and n!=other_anchor_node]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
            # parents of anchor node
            else:
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node) if n != query_node and n!=other_anchor_node]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                # siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if
                                n != self.pseudo_leaf_node and n != query_node and n!=other_anchor_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)), k=self.expand_factor) if
                                n != self.pseudo_leaf_node and n != query_node and n!=other_anchor_node]
                nodes.extend(siblings)
                nodes_pos.extend([2] * len(siblings))

        # create dgl graph with features
        g = dgl.DGLGraph()
        g.add_nodes(len(nodes), {"_id": torch.tensor(nodes), "pos": torch.tensor(nodes_pos)})

        # add edge types
        length = len(list(range(parent_node_idx)))
        g.add_edges(list(range(parent_node_idx)), parent_node_idx, {'erel': torch.tensor([0] * length)})

        length = len(list(range(parent_node_idx + 1, len(nodes))))
        g.add_edges(parent_node_idx, list(range(parent_node_idx + 1, len(nodes))), {'erel': torch.tensor([1] * length)})

        # add self-cycle
        g.add_edges(g.nodes(), g.nodes(), {'erel': torch.tensor([2] * len(g.nodes()))})

        return g


class PathDataset(RawDataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        super(PathDataset, self).__init__(graph_dataset, mode, sampling_mode, negative_size, max_pos_size,
                                           expand_factor, cache_refresh_time, normalize_embed, test_topk)
        self.node2root_path = self._get_path_to_root()
        self.node2leaf_path = self._get_path_to_leaf()

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        One data instance is a list of (anchor_egonet, query_node_feature, label) triplets.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
                res.append([u, v, u_path, v_path, lens, query_node, (1, 1, 1, 1)])
        elif self.sampling_mode > 0:
            u, v = random.choice(self.node2pos[query_node])
            u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
            res.append([u, v, u_path, v_path, lens, query_node, (1, 1, 1, 1)])

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u, v in negative_anchors:
            u_flag = int(u in self.node2parents[query_node])
            v_flag = int(v in self.node2children[query_node])
            e_flag = int(self.edge2dist[(u, v)] <= 2)
            u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
            res.append([u, v, u_path, v_path, lens, query_node, (0, u_flag, v_flag, e_flag)])

        return tuple(res)

    def _get_path_to_root(self):
        node2root_path = {n:[] for n in self.node_list}
        q = deque([self.root])
        node2root_path[self.root] = [[self.root]]
        visit = []
        while q:
            i = q.popleft()
            if i in visit:
                continue
            else:
                visit.append(i)
            children = self.core_subgraph.successors(i)
            for c in children:
                if c == self.pseudo_leaf_node:
                    continue
                if c not in q:
                    q.append(c)
                for path in node2root_path[i]:
                    node2root_path[c].append([c]+path)
        return node2root_path

    def _get_path_to_leaf(self):
        leafs = [n for n in self.core_subgraph.nodes if self.core_subgraph.out_degree(n)==1]
        node2leaf_path = {n:[] for n in self.node_list}
        q = deque(leafs)
        for n in leafs:
            node2leaf_path[n] = [[n, self.pseudo_leaf_node]]
        visit = []
        while q:
            i = q.popleft()
            if i in visit:
                continue
            else:
                visit.append(i)
            parents = self.core_subgraph.predecessors(i)
            for p in parents:
                if p == self.root:
                    continue
                if p not in q:
                    q.append(p)
                for path in node2leaf_path[i]:
                    node2leaf_path[p].append([p]+path)
        return node2leaf_path

    def _get_edge_node_path(self, query_node, edge):
        pu = random.choice(self.node2root_path[edge[0]])
        pu = [n for n in pu if n!=query_node]
        if edge[1] == self.pseudo_leaf_node:
            pv = [self.pseudo_leaf_node]
        else:
            pv = random.choice(self.node2leaf_path[edge[1]])
            pv = [n for n in pv if n!=query_node]
        len_pu = len(pu)
        len_pv = len(pv)
        return pu, pv, (len_pu, len_pv)

    def _get_batch_edge_node_path(self, edges):
        bpu, bpv, lens = zip(*[self._get_edge_node_path(None, edge) for edge in edges])
        lens = torch.tensor(lens)
        max_u, max_v = lens.max(dim=0)[0]
        bpu = [p+[self.pseudo_leaf_node]*(max_u-len(p)) for p in bpu]
        bpv = [p+[self.pseudo_leaf_node]*(max_v-len(p)) for p in bpv]
        return torch.tensor(bpu), torch.tensor(bpv), lens


class GraphPathDataset(GraphDataset, PathDataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        super(GraphPathDataset, self).__init__(graph_dataset, mode, sampling_mode, negative_size, max_pos_size,
                                          expand_factor, cache_refresh_time, normalize_embed, test_topk)

    def __getitem__(self, idx):
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
                u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
                res.append([u, v, u_egonet, v_egonet, u_path, v_path, lens, query_node, (1, 1, 1, 1)])
        elif self.sampling_mode > 0:
            u, v = random.choice(self.node2pos[query_node])
            u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
            u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
            res.append([u, v, u_egonet, v_egonet, u_path, v_path, lens, query_node, (1, 1, 1, 1)])

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u, v in negative_anchors:
            u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
            u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
            u_flag = int(u in self.node2parents[query_node])
            v_flag = int(v in self.node2children[query_node])
            e_flag = int(self.edge2dist[(u, v)] <= 2)
            res.append([u, v, u_egonet, v_egonet, u_path, v_path, lens, query_node, (0, u_flag, v_flag, e_flag)])

        return tuple(res)


class ExpanDataset(GraphPathDataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, 
                max_pos_size=100, expand_factor=64, cache_refresh_time=128, 
                normalize_embed=False, test_topk=-1, split_in_oov=False,
                negative_parent=False):
        start = time.time()
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed
        self.test_topk = test_topk
        self.split_in_oov = split_in_oov
        # New whether sample from another direction
        self.negative_parent = negative_parent
        if graph_dataset.candidates_separate_available:
            self.candidates_specific = graph_dataset.candidates_node_ids

        self.tx_id2node_id = graph_dataset.tx_id2node_id
        self.node_id2tx_id = graph_dataset.node_id2tx_id
        self.tx_id2taxon = graph_dataset.tx_id2taxon
        self.taxonomy = graph_dataset.taxonomy

        self.node_features = graph_dataset.g_full.ndata['x']
        full_graph = graph_dataset.g_full.to_networkx()
        train_node_ids = graph_dataset.train_node_ids
        roots = [node for node in full_graph.nodes() if full_graph.in_degree(node) == 0]
        if len(roots) > 1:
            self.root = len(full_graph.nodes)
            for r in roots:
                full_graph.add_edge(self.root, r)
            root_vector = torch.mean(self.node_features[roots], dim=0, keepdim=True)
            self.node_features = torch.cat((self.node_features, root_vector), 0)
            self.vocab = graph_dataset.vocab + ['root', 'leaf']
            train_node_ids.append(self.root)
        else:
            self.root = roots[0]
            self.vocab = graph_dataset.vocab + ['leaf']
        self.full_graph = full_graph

        # get some useful arguments from graph_dataset
        self.vocab_level = graph_dataset.vocab_level
        self.max_level = graph_dataset.max_level
        print("vocab_level count")
        values, counts = np.unique(self.vocab_level, return_counts=True)
        print(values)
        print(counts)

        # add pseudo leaf node to core graph
        self.core_subgraph = self._get_holdout_subgraph(train_node_ids)
        self.pseudo_leaf_node = len(full_graph.nodes)
        # removed pesudo leaf nodes
        # for node in list(self.core_subgraph.nodes()):
        #     self.core_subgraph.add_edge(node, self.pseudo_leaf_node)
        self.leaf_nodes = [node for node in self.core_subgraph.nodes() if self.core_subgraph.out_degree(node) == 1]
        # for pseudo leaf node
        leaf_vector = torch.zeros((1, self.node_features.size(1)))  # zero vector works best
        self.node_features = torch.cat((self.node_features, leaf_vector), 0)
        if self.normalize_embed:
            self.node_features = F.normalize(self.node_features, p=2, dim=1)

        # add interested node list and subgraph
        # remove supersource nodes (i.e., nodes without in-degree 0)
        interested_node_set = set(train_node_ids) - set([self.root])
        self.node_list = list(interested_node_set)

        # build node2pos, node2nbs, node2edge
        self.node2pos = {}
        self.node2parents, self.node2children, self.node2nbs = {}, {}, {self.pseudo_leaf_node: []}
        for node in interested_node_set:
            parents = set(self.core_subgraph.predecessors(node))
            children = set(self.core_subgraph.successors(node))
            if len(children) > 1:
                children = [i for i in children if i != self.pseudo_leaf_node]
            self.node2pos[node] = list(parents)
            self.node2parents[node] = parents
            self.node2children[node] = children
            self.node2nbs[node] = parents.union(children)

        holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.validation_node_ids)
        
        if self.split_in_oov:
            # sample 500 in voc nodes that are in the training set already, add to valid set as in-voc valid set
            self.valid_node_list_in = random.sample(train_node_ids, 500)
            valid_node2pos = self._find_insert_posistion(graph_dataset.validation_node_ids + self.valid_node_list_in, holdout_subgraph)
        else:
            valid_node2pos = self._find_insert_posistion(graph_dataset.validation_node_ids, holdout_subgraph)
        
        print(len(valid_node2pos.items())) # 1500
        # print(list(valid_node2pos.items())[:10])

        self.valid_node2pos = {node: set([p for (p, c) in pos_l if c == self.pseudo_leaf_node]) for node, pos_l in valid_node2pos.items()}
        print(len(self.valid_node2pos.items())) # 1500
        # print(list(self.valid_node2pos.items())[:10])
        self.valid_node2parents = {node: set([p for (p, c) in pos_l]) for node, pos_l in valid_node2pos.items()}
        self.valid_node_list = [node for node, pos in self.valid_node2pos.items() if len(pos)]

        if self.split_in_oov:
            # self.valid_node_list is all nodes in valid set
            # self.valid_node_list_in is the in-voc words in valid set
            # self.valid_node_list_out is the oov words in the valid set
            self.valid_node_list_out = list(set(self.valid_node_list)-set(self.valid_node_list_in))

            # create oov flag list for all node in the self.valid_node_list
            # 0: not oov, 1: oov
            self.valid_node_list_oov_flags = [1 if n in self.valid_node_list_out else 0 for n in self.valid_node_list]

            print(len(graph_dataset.train_node_ids)) # 11937
            print(len(graph_dataset.validation_node_ids)) #1000
            print(len(self.valid_node_list)) #1353
            print(len(self.valid_node_list_in)) # 500 
            print(len(self.valid_node_list_out)) # 1000
            print(len(self.valid_node_list_oov_flags)) #1353

        holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids)
        test_node2pos = self._find_insert_posistion(graph_dataset.test_node_ids, holdout_subgraph)
        self.test_node2pos = {node: set([p for (p, c) in pos_l if c == self.pseudo_leaf_node]) for node, pos_l in test_node2pos.items()}
        self.test_node2parent = {node: set([p for (p, c) in pos_l]) for node, pos_l in test_node2pos.items()}
        self.test_node_list = [node for node, pos in self.test_node2pos.items() if len(pos)]

        # used for sampling negative positions during train/validation stage
        self.pointer = 0
        self.all_nodes = list(self.core_subgraph.nodes())
        random.shuffle(self.all_nodes)

        # used for caching local subgraphs
        self.cache = {}  # if g = self.cache[anchor_node], then g is the egonet centered on the anchor_node
        self.cache_counter = {}  # if n = self.cache[anchor_node], then n is the number of times you used this cache

        self.node2root_path = self._get_path_to_root()

        if self.split_in_oov:
            # new: create word list for all in voc words (across train/valid/test sets)
            self.in_voc_dict = list(set(self.all_nodes)-set(self.valid_node_list_out)-set(self.test_node_list))

        # used for sampling negative positions during train/validation stage
        # all_edges was missing, since this Class is not inherited from RawDataset
        # so we add this all_edges argument
        self.all_edges = list(self._get_candidate_positions(self.core_subgraph))

        end = time.time()
        print(f"Finish loading dataset ({end - start} seconds)")

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        One data instance is a list of (anchor_egonet, query_node_feature, label) triplets.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u in pos_positions:
                u_egonet = self._get_subgraph_and_node_pair(query_node, u)
                u_path, lens = self._get_edge_node_path(query_node, u)
                res.append([u, u_egonet, u_path, lens, query_node, 1])

            if len(pos_positions) > 0:
                query_node_parent = pos_positions[0]
        elif self.sampling_mode > 0:
            u = random.choice(self.node2pos[query_node])
            u_egonet = self._get_subgraph_and_node_pair(query_node, u)
            u_path, lens = self._get_edge_node_path(query_node, u)
            res.append([u, u_egonet, u_path, lens, query_node, 1])

            query_node_parent = u

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u in negative_anchors:
            u_egonet = self._get_subgraph_and_node_pair(query_node, u)
            u_path, lens = self._get_edge_node_path(query_node, u)
            res.append([u, u_egonet, u_path, lens, query_node, 0])

        # select negative children
        if self.negative_parent:
            negative_anchors = self._get_negative_anchors(query_node_parent, int(np.floor(1*negative_size)), node2children=True)
            # generate negative triplets
            for u in negative_anchors:
                q_egonet = self._get_subgraph_and_node_pair(u, query_node_parent)
                q_path, lens = self._get_edge_node_path(u, query_node_parent)
                res.append([query_node_parent, q_egonet, q_path, lens, u, 0])

        return tuple(res)

    def _get_negative_anchors(self, query_node, negative_size, node2children=False):
        if self.sampling_mode == 0:
            return self._get_at_most_k_negatives(query_node, negative_size, node2children=node2children)
        elif self.sampling_mode == 1:
            return self._get_exactly_k_negatives(query_node, negative_size, node2children=node2children)

    def _get_at_most_k_negatives(self, query_node, negative_size, node2children=False):
        """ Generate AT MOST negative_size samples for the query node
        Args:
            node2children: if True, sample negatives fix the parent query
        """
        if self.pointer == 0:
            random.shuffle(self.all_nodes)

        while True:
            if not node2children:
                negatives = [ele for ele in self.all_nodes[self.pointer: self.pointer + negative_size] if
                         ele not in self.node2pos[query_node]]
            else:
                # handle the case that node2children do not have query_node when sampling from parent to child
                if query_node in self.node2children:
                    children_list = list(self.node2children[query_node])
                else:
                    children_list = []
                negatives = [ele for ele in self.all_nodes[self.pointer: self.pointer + negative_size] if
                         ele not in children_list]
            if len(negatives) > 0:
                break
            self.pointer += negative_size
            if self.pointer >= len(self.all_nodes):
                self.pointer = 0

        return negatives

    def _get_exactly_k_negatives(self, query_node, negative_size, node2children=False, ignore=[]):
        """ Generate EXACTLY negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.all_nodes)

        negatives = []
        while len(negatives) != negative_size:
            n_lack = negative_size - len(negatives)
            if not node2children:
                negatives.extend([ele for ele in self.all_nodes[self.pointer: self.pointer + n_lack] if
                                ele not in self.node2pos[query_node] and ele not in ignore])
            else:
                # handle the case that node2children do not have query_node when sampling from parent to child
                if query_node in self.node2children:
                    children_list = list(self.node2children[query_node])
                else:
                    children_list = []
                negatives.extend([ele for ele in self.all_nodes[self.pointer: self.pointer + n_lack] if
                                ele not in children_list and ele not in ignore])
            self.pointer += n_lack
            if self.pointer >= len(self.all_nodes):
                self.pointer = 0
                random.shuffle(self.all_nodes)
        if len(negatives) > negative_size:
            negatives = negatives[:negative_size]

        return negatives

    def _get_subgraph_and_node_pair(self, query_node, anchor_node_u):
        """ Generate anchor_egonet and obtain query_node feature

        instance_mode: 0 means negative example, 1 means positive example
        """

        # [IMPORTANT]
        cache_flag = self._check_cache_flag(anchor_node_u)
        # handle the case that node2nbs do not have query_node when sampling from parent to child
        if query_node in self.node2nbs:
            nbs_list = self.node2nbs[query_node]
        else:
            nbs_list = []
        flag = (query_node < 0) or (anchor_node_u not in nbs_list)
        if flag and cache_flag:
            g_u = self.cache[anchor_node_u]
            # self.cache_counter[anchor_node_u] += 1
        else:
            g_u = self._get_subgraph(query_node, anchor_node_u, flag)
            if flag:  # save to cache
                self.cache[anchor_node_u] = g_u
                self.cache_counter[anchor_node_u] = 0

        return g_u

    def _get_subgraph(self, query_node, anchor_node, instance_mode):
        one_hop = True
        if instance_mode:  # do not need to worry about query_node appears to be the child of anchor_node
            # parents of anchor node
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor)]
                nodes_pos = [0] * len(nodes)
                nodes_dist = [1] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                nodes_dist.append(0)

                # # anchor node itself
                # parent_node_idx = 0
                # nodes = [anchor_node]
                # nodes_pos = [1]

            else:
                # 1. get direct parent
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node)]
                nodes_pos = [0] * len(nodes)
                nodes_dist = [1] * len(nodes)
                if not one_hop:
                    # 6. children of direct parent
                    for node_sibling in nodes:
                        if self.core_subgraph.out_degree(node_sibling) <= self.expand_factor:
                            siblings_2 = [n for n in self.core_subgraph.successors(node_sibling) if n != self.pseudo_leaf_node]
                        else:
                            siblings_2 = [n for n in
                                        random.choices(list(self.core_subgraph.successors(node_sibling)), k=self.expand_factor)
                                        if
                                        n != self.pseudo_leaf_node]
                        nodes.extend(siblings_2)
                        nodes_pos.extend([5] * len(siblings_2))
                        nodes_dist.extend([2] * len(siblings_2))
                    # 2. get ancestors, new!
                    # do not include anchor node and its direct parent in ancestors list
                    # do not include root node (first item)
                    ancestors = get_ancestors(self.core_subgraph, anchor_node)[1:-2]
                    ancestors.reverse() # closest ancestors first
                    # print([n.display_name for n in ancestors])
                    nodes.extend(ancestors)
                    nodes_pos.extend([3] * len(ancestors))
                    nodes_dist.extend(list(range(2, 2 + len(ancestors))))
                # 3. anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                nodes_dist.append(0)
                # 4. siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if n != self.pseudo_leaf_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)), k=self.expand_factor)
                                if
                                n != self.pseudo_leaf_node]
                nodes.extend(siblings)
                nodes_pos.extend([2] * len(siblings))
                nodes_dist.extend([1] * len(siblings))
                if not one_hop:
                    # 5. children of siblings
                    for node_sibling in siblings:
                        if self.core_subgraph.out_degree(node_sibling) <= self.expand_factor:
                            siblings_2 = [n for n in self.core_subgraph.successors(node_sibling) if n != self.pseudo_leaf_node]
                        else:
                            siblings_2 = [n for n in
                                        random.choices(list(self.core_subgraph.successors(node_sibling)), k=self.expand_factor)
                                        if
                                        n != self.pseudo_leaf_node]
                        nodes.extend(siblings_2)
                        nodes_pos.extend([4] * len(siblings_2))
                        nodes_dist.extend([2] * len(siblings_2))
        else:  # remove query_node from the children set of anchor_node
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor) if n != query_node]
                nodes_pos = [0] * len(nodes)
                nodes_dist = [1] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                nodes_dist.append(0)
                # parent_node_idx = 0
                # nodes = [anchor_node]
                # nodes_pos = [1]
            # parents of anchor node
            else:
                # 1. get direct parent
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node) if n != query_node]
                nodes_pos = [0] * len(nodes)
                nodes_dist = [1] * len(nodes)
                if not one_hop:
                    # 6. children of direct parent
                    for node_sibling in nodes:
                        if self.core_subgraph.out_degree(node_sibling) <= self.expand_factor:
                            siblings_2 = [n for n in self.core_subgraph.successors(node_sibling) if n != self.pseudo_leaf_node]
                        else:
                            siblings_2 = [n for n in
                                        random.choices(list(self.core_subgraph.successors(node_sibling)), k=self.expand_factor)
                                        if
                                        n != self.pseudo_leaf_node]
                        nodes.extend(siblings_2)
                        nodes_pos.extend([5] * len(siblings_2))
                        nodes_dist.extend([2] * len(siblings_2))
                    # 2. get ancestors, new!
                    # do not include anchor node and its direct parent in ancestors list
                    # do not include root node (first item)
                    ancestors = get_ancestors(self.core_subgraph, anchor_node)[1:-2] 
                    ancestors.reverse() # closest ancestors first
                    # print([n.display_name for n in ancestors])
                    nodes.extend(ancestors)
                    nodes_pos.extend([3] * len(ancestors))
                    nodes_dist.extend(list(range(2, 2 + len(ancestors))))
                # 3. anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                nodes_dist.append(0)
                # 4. siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if
                                n != self.pseudo_leaf_node and n != query_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)), k=self.expand_factor)
                                if
                                n != self.pseudo_leaf_node and n != query_node]
                nodes.extend(siblings)
                nodes_pos.extend([2] * len(siblings))
                nodes_dist.extend([1] * len(siblings))
                if not one_hop:
                    # 5. children of siblings
                    for node_sibling in siblings:
                        if self.core_subgraph.out_degree(node_sibling) <= self.expand_factor:
                            siblings_2 = [n for n in self.core_subgraph.successors(node_sibling) if n != self.pseudo_leaf_node]
                        else:
                            siblings_2 = [n for n in
                                        random.choices(list(self.core_subgraph.successors(node_sibling)), k=self.expand_factor)
                                        if
                                        n != self.pseudo_leaf_node]
                        nodes.extend(siblings_2)
                        nodes_pos.extend([4] * len(siblings_2))
                        nodes_dist.extend([2] * len(siblings_2))

        # removed pesudo leaf nodes
        # TODO: should we do this?
        nodes = [n for n in nodes if n != self.pseudo_leaf_node]

        # t1 = nodes[0]
        # print(t1)
        # print(self.vocab[t1])
        # if self.vocab[t1] != 'root':
        #     t2 = self.node_id2tx_id[t1]
        #     print(t2)
        #     t3 = self.tx_id2taxon[t2]
        #     print(t3)

        nodes_level = [self.tx_id2taxon[self.node_id2tx_id[n]].level+1 if self.vocab[n] != 'root' else 0 for n in nodes]
        # if it's root node, level is 0
        # for other nodes, add 1 to level

        # create dgl graph with features
        g = dgl.DGLGraph()
        g.add_nodes(len(nodes), {"_id": torch.tensor(nodes), 
                                "pos": torch.tensor(nodes_pos),
                                "dist": torch.tensor(nodes_dist), 
                                "abs_level": torch.tensor(nodes_level)})
        add_edge_for_dgl(g, list(range(parent_node_idx)), parent_node_idx)
        add_edge_for_dgl(g, parent_node_idx, list(range(parent_node_idx + 1, len(nodes))))

        # add self-cycle
        g.add_edges(g.nodes(), g.nodes())

        return g

    def _get_edge_node_path(self, query_node, parent):
        if parent == self.pseudo_leaf_node:
            pu = [self.pseudo_leaf_node]
        else:
            pu = random.choice(self.node2root_path[parent])
            pu = [n for n in pu if n!=query_node]
        len_pu = len(pu)
        return pu, len_pu

    def _get_batch_edge_node_path(self, edges):
        bpu, lens = zip(*[self._get_edge_node_path(None, edge) for edge in edges])
        lens = torch.tensor(lens)
        max_u = lens.max(dim=0)[0]
        bpu = [p+[self.pseudo_leaf_node]*(max_u-len(p)) for p in bpu]
        return torch.tensor(bpu), lens

def assemble_sentence_single(word_p, context_sens, passin_real_word=True, allow_empty=False, show=False):
    """
    Get exemple sentence for a word
    Args:
        word_p: the input word
        passin_real_word: (bool) whether the pass in parameter word_p is a strong (a real word)
                                or an integer index (that need to be checked from self.voc
                                )
        allow_empty: (bool) whether allow return empty string

    Returns:
        ex_p: the exemple sentence of word_p
        ex_p_before_main_lemma: partial sentence that cut before the main lemma (word_p)
        ex_p_contain_main_lemma: partial sentence that cut after the main lemma (word_p) so it contains word_p
    """
    # if not passin_real_word:
    #     word_p = self.voc[word_p]
    not_found_key_flag = False
    try:
        ex_p, df_p, main_lemma_p, ex_p_before_main_lemma, ex_p_contain_main_lemma = get_synset_related_sentence(word_p, context_sens, show=show)
    except:
        # print(f'-> No key found for {word_p}')
        ex_p = get_verb_from_id(word_p, source='wn') # for example, not found key train.withdef.96
        not_found_key_flag = True

    if not_found_key_flag:
        ex_p = get_verb_from_id(word_p, source='wn')
        ex_p_before_main_lemma = ''
        ex_p_contain_main_lemma = ex_p

    return ex_p, ex_p_before_main_lemma, ex_p_contain_main_lemma, not_found_key_flag

def get_synset_related_sentence(synset_id, context_sens, combine_sources=True, show=False):
    example_list = wn.synset(synset_id).examples()
    definition_str = wn.synset(synset_id).definition()
    example_str = '. '.join(example_list).lower()
    
    if combine_sources:
        example_str = example_str + '. ' + definition_str + '. ' + '. '.join(context_sens)

    # lemmatize example_str
    example_str_lemmatize = ' '.join([wnl.lemmatize(i,j[0].lower()).lower() if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i).lower() for i,j in pos_tag(word_tokenize(example_str))])

    # process example sentence to replace the lemma with the synset main lemma here
    lemmas = [str(lemma.name()) for lemma in wn.synset(synset_id).lemmas()]
    main_lemma = get_verb_from_id(synset_id, source='wn')
    # print(synset_id, lemmas, main_lemma)
    example_str_tokenized = word_tokenize(example_str)
    example_str_before_main_lemma = []
    example_str_contain_main_lemma = example_str_tokenized[0] if len(example_str_tokenized) > 0 else []
    example_str_lemmatize_tokenized = word_tokenize(example_str_lemmatize)
    for lemma in lemmas:
        lemma_this = lemma.replace('_', ' ').lower()
        # if lemma_this in example_str:
        #     example_str = example_str.replace(lemma_this, main_lemma)
        indices = [i for i, x in enumerate(example_str_lemmatize_tokenized) if x == lemma_this]
        if len(indices) > 0:
            for i in indices:
                # if example_str_lemmatize_tokenized[i] != main_lemma:
                example_str_tokenized[i] = main_lemma
                example_str_before_main_lemma = example_str_tokenized[:i]
                example_str_contain_main_lemma = example_str_tokenized[:i+1]
            
    example_str = ' '.join(example_str_tokenized)
    example_str_before_main_lemma = ' '.join(example_str_before_main_lemma)
    example_str_contain_main_lemma = ' '.join(example_str_contain_main_lemma)

    if isinstance(example_str_contain_main_lemma, str):
        example_str_contain_main_lemma = example_str_contain_main_lemma
    else:
        example_str_contain_main_lemma = ' '.join(example_str_contain_main_lemma)

    return example_str, definition_str, main_lemma, example_str_before_main_lemma, example_str_contain_main_lemma

def get_wn_id_from_str(node_str):
    if node_str in ['root', 'leaf']:
        return node_str
    else:
        return node_str.split('||')[1].split('@@@')[0]