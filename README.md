# HyperExpan: Taxonomy Expansion with Hyperbolic Representation Learning

Source code for paper [HyperExpan: Taxonomy Expansion with Hyperbolic Representation Learning](https://aclanthology.org/2021.findings-emnlp.353) in Findings of ACL: EMNLP 2021.

[[PDF]](https://aclanthology.org/2021.findings-emnlp.353.pdf) [[Talk]](https://youtu.be/zUGIunzjfVE) [[Slides]](https://derek.ma/pubs/ma-etal-2021-hyperexpan-taxonomy/slides_EMNLP21.pdf) [[Cite]](#cite)

## Dependencies

1. Create new conda environemnt

```
conda env create -f env.yml
```

2.  Install DGL 0.4.0 version with GPU suppert using Conda from following page: [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)

```
conda activate hyperexpan
conda install dgl-cuda11.1 -c dglteam
```

## Data Preparation

### Download processed data

For dataset used in our paper, you can directly download all input files below and skip this section. We generate the initial feature vectors following [TaxoExpan's implementation](https://github.com/mickeysjm/TaxoExpan#miscellaneous-notes--details).

* [**MAG-CS**](https://drive.google.com/file/d/11bxzqM8qznI-Qx1aAImd_JDBytw-_rjc/view?usp=sharing)
* [**MAG-Psy**](https://drive.google.com/file/d/1joeKI9qvtHl8VX9Dfs9uy2G3EUu0PLDs/view?usp=sharing)
* [**WordNet-Noun** ](https://drive.google.com/file/d/1S6ijwV7phg6ZlJbUgSZjPuJTcN98bWwe/view?usp=sharing)
* [**WordNet-Verb**](https://drive.google.com/file/d/13LqeaaPq6vS8ah-dgkJO2eWGp107eSfT/view?usp=sharing)

### New taxonomy

For expanding new input taxonomies, you need to read this section and format your datasets accordingly.

#### Step 0.a (Required): Organize your input taxonomy along with node features into the following 3 files

**1. <TAXONOMY_NAME>.terms**, each line represents one concept in the taxonomy, including its ID and surface name

```
taxon1_id \t taxon1_surface_name
taxon2_id \t taxon2_surface_name
taxon3_id \t taxon3_surface_name
...
```

**2. <TAXONOMY_NAME>.taxo**, each line represents one relation in the taxonomy, including the parent taxon ID and child taxon ID

```
parent_taxon1_id \t child_taxon1_id
parent_taxon2_id \t child_taxon2_id
parent_taxon3_id \t child_taxon3_id
...
```

**3. <TAXONOMY_NAME>.terms.<EMBED_SUFFIX>.embed**, the first line indicates the vocabulary size and embedding dimension, each of the following line represents one taxon with its pretrained embedding

```
<VOCAB_SIZE> <EMBED_DIM>
taxon1_id taxon1_embedding
taxon2_id taxon2_embedding
taxon3_id taxon3_embedding
...
```

The embedding file follows the gensim word2vec format.

Notes:

1. Make sure the <TAXONOMY_NAME> is the same across all the 3 files.
2. The <EMBED_SUFFIX> is used to chooose what initial embedding you will use. You can leave it empty to load the file "<TAXONOMY_NAME>.terms.embed". **Make sure you can generate the embedding for a new given term.**

#### Step 0.b (Optional): Generate train/validation/test partition files

You can generate your desired train/validation/test parition files by creating another 3 separated files (named <TAXONOMY_NAME>.terms.train, <TAXONOMY_NAME>.terms.validation, as well as <TAXONOMY_NAME>.terms.test) and puting them in the same directory as the above three required files.

These three partition files are of the same format -- each line includes one taxon_id that appears in the above <TAXONOMY_NAME>.terms file.

#### Step 1: Generate the binary dataset file

1. Create a folder "./data/{DATASET_NAME}"
2. Put the above three required files (as well as three optional partition files) in "./data/{DATASET_NAME}"
3. Under this root directory, run

```
python generate_dataset_binary.py \
    --taxon_name <TAXONOMY_NAME> \
    --data_dir <DATASET_NAME> \
    --embed_suffix <EMBED_SUFFIX> \
    --existing_partition 0 \
    --partition_pattern leaf \
```

Such as:
```
python generate_dataset_binary.py \
    --taxon_name wordnet_verb \
    --data_dir data/SemEval-Verb \
    --embed_suffix fasttext_mode4_expan \
    --existing_partition 0 \
    --partition_pattern leaf
```

This script will first load the existing taxonomy (along with initial node features indicated by `embed_suffix`) from the previous three required files.
Then, if `existing_partition` is 0, it will generate a random train/validation/test partitions, otherwise, it will load the existing train/validation/test partition files.
Notice that if `partition_pattern` is `internal`, it will randomly sample both internal and leaf nodes for validation/test, which makes it a taxonomy completion task; if it is set `leaf`, it will become a taxonomy expansion task.
Finally, it saves the generated dataset (along with all initial node features) in one pickle file for fast loading next time.

## Model Training and Evaluation

### Quick start

Start training and evaluation following the arguments set in a config file under `config`, this script will run both validation and testing along with training.

```
python train.py --config config/DATANAME.json
```

### Key arguments in config files

* `mode`:  `r`, `p` and `g`, representing initial embedding, LSTM and GNN respectively. If you want to use a combination of initial embedding and GNN encoder, plz set `mode` to `rg`, and then the initial embedding and embedding output by GNN encoder will be concatenated for calculating matching score.
* `type`: which model class to be used defined in `model/model.py`. Set it to `ExpanMatchModel` for taxonomy expansion, `MatchModel` for taxonomy completion.
* `trainer`: which trainer class to be used defined in `trainer/trainer.py`. 
* `hgcn_args/manifold`: which hyperbolic model to be used. Options are: `PoincareBall`, `Hyperboloid` or `Euclidean`.
* `train_data_loader/type`: which dataloader class to use, as defined in `data_loader/data_loaders.py`.
* `trainer/monitor`: which metric to monitor to save the best epoch, `max val_mrr_scaled_10` means select the checkpoint with highest validation `mrr_scaled_10` result.

## Acknowledgement

For all implementations, we follow the project organization in [pytorch-template](https://github.com/victoresque/pytorch-template). Part of this codebase is adapted from previous works [TMN](https://github.com/JieyuZ2/TMN) and [TaxoExpan](https://github.com/mickeysjm/TaxoExpan).

## Cite

```
@inproceedings{ma-etal-2021-hyperexpan-taxonomy,
    title = "{H}yper{E}xpan: Taxonomy Expansion with Hyperbolic Representation Learning",
    author = "Ma, Mingyu Derek  and
      Chen, Muhao  and
      Wu, Te-Lin  and
      Peng, Nanyun",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.353",
    doi = "10.18653/v1/2021.findings-emnlp.353",
    pages = "4182--4194",
    abstract = "Taxonomies are valuable resources for many applications, but the limited coverage due to the expensive manual curation process hinders their general applicability. Prior works attempt to automatically expand existing taxonomies to improve their coverage by learning concept embeddings in Euclidean space, while taxonomies, inherently hierarchical, more naturally align with the geometric properties of a hyperbolic space. In this paper, we present HyperExpan, a taxonomy expansion algorithm that seeks to preserve the structure of a taxonomy in a more expressive hyperbolic embedding space and learn to represent concepts and their relations with a Hyperbolic Graph Neural Network (HGNN). Specifically, HyperExpan leverages position embeddings to exploit the structure of the existing taxonomies, and characterizes the concept profile information to support the inference on new concepts that are unseen during training. Experiments show that our proposed HyperExpan outperforms baseline models with representation learning in a Euclidean feature space and achieves state-of-the-art performance on the taxonomy expansion benchmarks.",
}
```