# Resources

Saber is ready to go out-of-the box when using the __web-service__ or a __pre-trained model__. However, if you plan on training you own models, you will need to provide a dataset (or datasets!) and, ideally, pre-trained word embeddings.

## Pre-trained models

Pre-trained model names can be passed to `Saber.load()` (see [Quick Start: Pre-trained Models](https://baderlab.github.io/saber/quick_start/#pre-trained-models)). Appending `"*-large"` to the model name (e.g. `"PRGE-large"` will download a much larger model, which should perform slightly better than the base model.

Identifier | Semantic Group | Identified entity types | Namespace |
---------- | -------------- | ----------------------- | --------- |
`CHED` | Chemicals | Abbreviations and Acronyms, Molecular Formulas, Chemical database identifiers, IUPAC names, Trivial (common names of chemicals and trademark names), Family (chemical families with a defined structure) and Multiple (non-continuous mentions of chemicals in text) | [PubChem Compounds](https://pubchem.ncbi.nlm.nih.gov/)
`DISO` | Disorders | Acquired Abnormality, Anatomical Abnormality, Cell or Molecular Dysfunction, Congenital Abnormality, Disease or Syndrome, Mental or Behavioral Dysfunction, Neoplastic Process, Pathologic Function, Sign or Symptom | [Disease Ontology](http://disease-ontology.org/)
`LIVB` | Organisms | Species, Taxa | [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/taxonomy)
`PRGE` | Genes and Gene Products | Genes, Gene Products | [STRING](https://string-db.org/)

## Datasets

Currently, Saber requires corpora to be in a **CoNLL** format with a BIO or IOBES tag scheme, e.g.:

```
Selegiline	B-CHED
-	O
induced	O
postural	B-DISO
hypotension	I-DISO
...
```

Corpora in such a format are collected in [here](https://github.com/BaderLab/Biomedical-Corpora) for convenience.

!!! info
      Many of the corpora in the BIO and IOBES tag format were originally collected by [Crichton _et al_., 2017](https://doi.org/10.1186/s12859-017-1776-8), [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016).

In this format, the first column contains each token of an input sentence, the last column contains the tokens tag, all columns are separated by tabs, and all sentences by a newline.

Of course, not all corpora are distributed in the CoNLL format:

- Corpora in the **Standoff** format can be converted to **CoNLL** format using [this](https://github.com/spyysalo/standoff2conll) tool.
- Corpora in **PubTator** format can be converted to **Standoff** first using [this](https://github.com/spyysalo/pubtator) tool.

Saber infers the "training strategy" based on the structure of the dataset folder:

- To use k-fold cross-validation, simply provide a `train.*` file in your dataset folder.

E.g.
```
.
├── NCBI_Disease
│   └── train.tsv
```

- To use a train/valid/test strategy, provide `train.*` and `test.*` files in your dataset folder. Optionally, you can provide a `valid.*` file. If not provided, a random 10% of examples from `train.*` are used as the validation set.

E.g.
```
.
├── NCBI_Disease
│   ├── test.tsv
│   └── train.tsv
```

## Word embeddings

When training new models, you can (and should) provide your own pre-trained word embeddings with the `pretrained_embeddings` argument (either at the command line or in the configuration file). Saber expects all word embeddings to be in the `word2vec` file format. [Pyysalo _et al_. 2013](https://pdfs.semanticscholar.org/e2f2/8568031e1902d4f8ee818261f0f2c20de6dd.pdf) provide word embeddings that work quite well in the biomedical domain, which can be downloaded [here](http://bio.nlplab.org). Alternatively, from the command line call:

```
# Replace this with a location you want to save the embeddings to
$ mkdir path/to/word_embeddings
# Note: this file is over 4GB
$ wget http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin -O path/to/word_embeddings
```

To use these word embeddings with Saber, provide their path in the `pretrained_embeddings` argument (either in the `config` file or at the command line). Alternatively, pass their path to `Saber.load_embeddings()`. For example:

```python
from saber.saber import Saber

saber = Saber()

saber.load_dataset('path/to/dataset')
# load the embeddings here
saber.load_embeddings('path/to/word_embeddings')

saber.build()
saber.train()
```

### GloVe

To use [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings, just convert them to the [word2vec](https://code.google.com/archive/p/word2vec/) format first:

```
(saber) $ python
>>> from gensim.scripts.glove2word2vec import glove2word2vec
>>> glove_input_file = 'glove.txt'
>>> word2vec_output_file = 'word2vec.txt'
>>> glove2word2vec(glove_input_file, word2vec_output_file)
```
