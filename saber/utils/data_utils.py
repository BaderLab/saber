"""A collection of data-related helper/utility functions.
"""
import glob
import logging
import os
from itertools import chain

from sklearn.model_selection import KFold, train_test_split

from .. import constants
from ..preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)

def get_filepaths(filepath):
    """Returns dictionary with filepaths to `train`/`valid`/`test` partitions from `filepath`.

    Returns a dictionary containing filepaths to `train`, `valid` and `test` partitions at
    `filepath/train.*``, `filepath/valid.*` and `filepath/test.*`. A train set must be provided,
    while valid and test sets are optional.

    Args:
        filepath (str): Path to dataset folder.

    Returns:
        A dictionary with keys `train`, and optionally, `valid` and `test` containing the
        filepaths to the train, valid and test paritions of the dataset at `filepath`.

    Raises:
        ValueError when no file at `filepath/train.*` is found.
    """
    partition_filepaths = {}
    # search for partition filepaths
    train_partition = glob.glob(os.path.join(filepath, constants.TRAIN_FILE))
    valid_partition = glob.glob(os.path.join(filepath, constants.VALID_FILE))
    test_partition = glob.glob(os.path.join(filepath, constants.TEST_FILE))

    # must supply a train file
    if not train_partition:
        err_msg = "Must supply at least one file, train.* at {}".format(filepath)
        LOGGER.error('ValueError %s', err_msg)
        raise ValueError(err_msg)

    # collect filepaths in a dictionary, valid and test files are optional
    partition_filepaths['train'] = train_partition[0]
    partition_filepaths['valid'] = valid_partition[0] if valid_partition else None
    partition_filepaths['test'] = test_partition[0] if test_partition else None

    return partition_filepaths

def load_single_dataset(config):
    """Loads a single dataset.

    Creates and loads a single dataset object for a dataset at `config.dataset_folder[0]`.

    Args:
        config (Config): A Config object which contains a set of harmonized arguments provided in
            a *.ini file and, optionally, from the command line.

    Returns:
        A list containing a single Dataset object.
    """
    from ..dataset import Dataset # breaks circular import

    dataset = Dataset(directory=config.dataset_folder[0],
                      replace_rare_tokens=config.replace_rare_tokens)
    dataset.load()

    return [dataset]

def load_compound_dataset(config):
    """Loads a compound dataset.

    Creates and loads a `compound` dataset. Compound datasets are specified by multiple
    individual datasets, and share multiple attributes (such as `word` and `char` type to index
    mappings). Loads such a dataset for each dataset at `config.dataset_folder`.

    Args:
        config (Config): A Config object which contains a set of harmonized arguments provided in
            a *.ini file and, optionally, from the command line.

    Returns:
        A list containing multiple Dataset objects.
    """
    from ..dataset import Dataset # breaks circular import

    # accumulate and load each dataset
    compound_dataset = []
    for dir_ in config.dataset_folder:
        dataset = Dataset(directory=dir_, replace_rare_tokens=config.replace_rare_tokens)
        dataset.load()
        compound_dataset.append(dataset)

    # to generate a compound dataset, we need to:
    # 1. pool word and char types
    # 2. compute mappings of these pooled types to unique integer IDs
    # 3. update each datasets type_to_idx mappings (for word and char types only)
    # 4. re-compute the index sequences

    # 1. pool word and char types
    combined_types = {'word': [dataset.type_to_idx['word'] for dataset in compound_dataset],
                      'char': [dataset.type_to_idx['char'] for dataset in compound_dataset]}
    combined_types['word'] = list(set(chain.from_iterable(combined_types['word'])))
    combined_types['char'] = list(set(chain.from_iterable(combined_types['char'])))
    # 2. compute mappings of these pooled types to unique integer IDs
    type_to_idx = {
        'word': Preprocessor.type_to_idx(combined_types['word'], constants.INITIAL_MAPPING['word']),
        'char': Preprocessor.type_to_idx(combined_types['char'], constants.INITIAL_MAPPING['word']),
    }
    for dataset in compound_dataset:
        # 3. update each datasets type_to_idx mappings (for word and char types only)
        word_types, char_types = list(dataset.type_to_idx['word']), list(dataset.type_to_idx['char'])
        dataset.type_to_idx['word'] = Preprocessor.type_to_idx(word_types, type_to_idx['word'])
        dataset.type_to_idx['char'] = Preprocessor.type_to_idx(char_types, type_to_idx['char'])
        # 4. re-compute the index sequences
        dataset.get_idx_seq()

    return compound_dataset

def setup_dataset_for_transfer(dataset, type_to_idx):
    """Modifys a `Dataset` object when transfer learning.

    Performs a series of steps to a loaded Dataset object (`dataset`) so that it can be used as
    the target dataset when transfer learning. Namely, it replaces the `type_to_idx` mappings
    for words and characters with those of the source dataset and used them to re-generate
    `idx_seq`. This way, the target dataset contains only words and characters that appeared in the
    source dataset.

    Args:
        dataset (Dataset): A Dataset object for which `Dataset.load()` has been called.
        type_to_idx (dict): A dictionary mapping word, char and tag types to unique integer IDs.
    """
    # overwrite type to index maps
    dataset.type_to_idx['word'] = type_to_idx['word']
    dataset.type_to_idx['char'] = type_to_idx['char']
    # re-generate index sequence
    dataset.get_idx_seq()

def collect_valid_data(training_data, test_size=0.10):
    """Splits training data (`training_data`) into train and validation partitions.

    Generates a new training set of size 1 - `test_size` and a validation set of size `test_size`
    from the original training data given in `training_data`. Expects `training_data` to be a
    dictionary with the keys 'x_train' and 'y_train', containing the input examples and targets
    respectively.

    Args:
        training_data (list): A list containing training data (inputs and targets) for each dataset.
        test_size (float): Percentage of the training examples in `training_data` to use for the new
            validation set.

    Returns:
        `training_data`, containing new train and validation sets at keys 'x_train'/'y_train' and
            'x_valid'/'y_valid' respectively.

    Raises:
        ValueError if `training_data` does not contain the keys 'x_train', 'y_train'.
    """
    if any(['x_train' not in data or 'y_train' not in data for data in training_data]):
        err_msg = "Argument `training_data` must contain the keys 'x_train' and 'y_train'"
        LOGGER.error("ValueError: %s", err_msg)
        raise ValueError(err_msg)

    for i, data in enumerate(training_data):
        x_train_word, x_valid_word, x_train_char, x_valid_char, y_train, y_valid = \
            train_test_split(data['x_train'][0],
                             data['x_train'][1],
                             data['y_train'],
                             test_size=test_size,
                             random_state=42,
                             shuffle=False)

        training_data[i] = {'x_train': [x_train_word, x_train_char],
                            'x_valid': [x_valid_word, x_valid_char],
                            'y_train': y_train,
                            'y_valid': y_valid,
                            # add back in test data
                            'x_test': data['x_test'],
                            'y_test': data['y_test'],
                           }

    return training_data

def get_train_valid_indices(training_data, k_folds):
    """Get `k_folds` number of sets of train/valid indices for all datasets in `datasets`.

    For all Dataset objects in `datasets`, gets `k_folds` number of train and valid indices. Returns
    a list of lists of two-tuples, where the outer list is of length `len(datasets)` and the inner
    list is of length `k_folds` and contains two-tuples corresponding to train and valid indices
    respectively. For example, the train indices for the ith dataset and jth fold would be
    train_valid_indices[i][j][0].

    Args:
        datasets (list): a list of Dataset objects.
        k_folds (int): Number of folds to compute indices for.

    Returns:
        A list of lists of two-tuples, where index [i][j] is a tuple containing the train and valid
        indicies (in that order) for the ith dataset and jth k-fold.
    """
    train_valid_indices = [] # acc
    kf = KFold(n_splits=k_folds, random_state=42) # Sklearn KFold object

    for i, _ in enumerate(training_data):
        X, _ = training_data[i]['x_train']
        train_valid_indices.append([(ti, vi) for ti, vi in kf.split(X)])

    return train_valid_indices

def get_data_partitions(training_data, train_valid_indices):
    """Get train and valid partitions for all k-folds for all datasets.

    For all Dataset objects in `datasets`, gets the train and valid partitions for the current
    k-fold (`fold`) using the indices given at `train_valid_indices`. Returns a list of lists
    of four-tuples: (x_train, x_valid, y_train, y_valid), where index i, j contains the data
    for the ith dataset and jth k-fold.

    Args:
        datasets (list): A list of Dataset objects.
        train_valid_indices (list): A list of list of two-tuples, where train_valid_indices[i][j].
            is a tuple containing the train and valid indices (in that order) for the ith dataset.
            and jth fold.
        fold (int): The current fold in k-fold cross-validation.

    Returns:
        A list of lists, `partitioned_data`, where `partitioned_data[i][j]` contains the data for
        the ith dataset and jth fold.
    """
    partitioned_data = []
    for i, _ in enumerate(train_valid_indices): # loop over datasets
        partitioned_data.append([])
        for j, _ in enumerate(train_valid_indices[i]): # loop over folds
            # train_valid_indices[i][j] is a two-tuple which contains the train and valid indices
            train_indices, valid_indices = train_valid_indices[i][j]
            # get inputs and targets for dataset i
            x_word_train, x_char_train = training_data[i]['x_train']
            y_train = training_data[i]['y_train']

            # create and accumulate train/valid partitions
            x_word_train, x_word_valid = x_word_train[train_indices], x_word_train[valid_indices]
            x_char_train, x_char_valid = x_char_train[train_indices], x_char_train[valid_indices]
            y_train, y_valid = y_train[train_indices], y_train[valid_indices]

            partitioned_data[i].append({'x_train': [x_word_train, x_char_train],
                                        'x_valid': [x_word_valid, x_char_valid],
                                        'y_train': y_train,
                                        'y_valid': y_valid,
                                        # add back in test data
                                        'x_test': training_data[i]['x_test'],
                                        'y_test': training_data[i]['y_test'],})

    return partitioned_data

def collect_cv_data(training_data, k_folds):
    """Splits training data (`training_data`) into `k_folds` number of train/valid partitions.

    Generates a train/valid split for `k_folds` number of cross-validation folds from the data
    given in `training_data`. Returns a list of lists, `partitioned_data`, where
    `partitioned_data[i][j]` contains the data for the ith dataset and jth fold.

    Args:
        training_data (list): A list containing training data (inputs and targets) for each dataset.
        k_folds (int): Number of folds to split the data into.

    Returns:
        A list of lists, `partitioned_data`, where `partitioned_data[i][j]` contains the data for
        the ith dataset and jth fold.
    """
    train_valid_indices = get_train_valid_indices(training_data, k_folds)
    partitioned_data = get_data_partitions(training_data, train_valid_indices)

    return partitioned_data
