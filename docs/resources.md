# Resources

Saber is ready to go out-of-the box when using the __web-service__ or a __pre-trained model__. However, if you plan on training you own models, you will need to provide a dataset (or datasets!).

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

### Partitioning Datasets for Evaluation

Saber infers the _partitioning strategy_ based on the structure of the dataset folder and the arguments `k_folds`, `validation_split`.

- To use k-fold cross-validation, provide only a `train.*` file under `dataset_folder`. Choose the number of folds with the `k_folds` argument. Optionally, specify a proportion of examples to hold-out at random in each fold as a validation set with `validation_split`.
- To create a validation split from the train set, provide a `train.*` file (and optionally, a `test.*` file) under `dataset_folder` and specify the proportion of training examples to hold-out for a validation set with `validation_split`.

Otherwise, provide the partitions yourself with the files `train.*`, `valid.*` and `test.*` under `dataset_folder` and leave `k_folds` and `validation_split` equal to `0`.

E.g.

```bash
.
├── NCBI_Disease
│   └── train.tsv
│   └── valid.tsv
│   └── test.tsv
```

!!! note
      `k_folds` will be ignored if either a `valid.*` or `test.*` file is found under `dataset_folder`. Both arguments `k_folds` and `validation_split` will be ignored if a `valid.*` file is found under `dataset_folder`. 