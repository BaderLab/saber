"""A collection of data-related helper/utility functions.
"""
import copy
import glob
import logging
import os
import time
from itertools import chain

from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

from ..constants import INITIAL_MAPPING
from ..constants import RANDOM_STATE
from ..constants import TEST_FILE
from ..constants import TRAIN_FILE
from ..constants import VALID_FILE
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

    train_partition = glob.glob(os.path.join(filepath, TRAIN_FILE))
    valid_partition = glob.glob(os.path.join(filepath, VALID_FILE))
    test_partition = glob.glob(os.path.join(filepath, TEST_FILE))

    if not train_partition:
        err_msg = "Must supply at least one file, train.* at {}".format(filepath)
        LOGGER.error('ValueError %s', err_msg)
        raise ValueError(err_msg)

    # Collect filepaths in a dictionary, valid and test files are optional
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
    if config.dataset_reader == 'conll2003datasetreader':
        from ..dataset import CoNLL2003DatasetReader
        dataset = CoNLL2003DatasetReader(dataset_folder=config.dataset_folder[0],
                                         replace_rare_tokens=config.replace_rare_tokens)
    elif config.dataset_reader == 'conll2004datasetreader':
        from ..dataset import CoNLL2004DatasetReader
        dataset = CoNLL2004DatasetReader(dataset_folder=config.dataset_folder[0],
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
    # accumulate and load each dataset
    compound_dataset = []
    for dir_ in config.dataset_folder:
        if config.dataset_reader == 'conll2003datasetreader':
            from ..dataset import CoNLL2003DatasetReader
            dataset = CoNLL2003DatasetReader(dataset_folder=dir_,
                                             replace_rare_tokens=config.replace_rare_tokens)
        elif config.dataset_reader == 'conll2004datasetreader':
            from ..dataset import CoNLL2004DatasetReader
            dataset = CoNLL2004DatasetReader(dataset_folder=dir_,
                                             replace_rare_tokens=config.replace_rare_tokens)
        dataset.load()
        compound_dataset.append(dataset)

    # To generate a compound dataset, we need to:
    # 1. Pool word and char types
    # 2. Compute mappings of these pooled types to unique integer IDs
    # 3. Update each datasets type_to_idx mappings (for word and char types only)
    # 4. Re-compute the index sequences

    # 1. Pool word and char types
    combined_types = {'word': [dataset.type_to_idx['word'] for dataset in compound_dataset],
                      'char': [dataset.type_to_idx['char'] for dataset in compound_dataset]}
    combined_types['word'] = list(set(chain.from_iterable(combined_types['word'])))
    combined_types['char'] = list(set(chain.from_iterable(combined_types['char'])))
    # 2. Compute mappings of these pooled types to unique integer IDs
    type_to_idx = {
        'word': Preprocessor.type_to_idx(combined_types['word'], INITIAL_MAPPING['word']),
        'char': Preprocessor.type_to_idx(combined_types['char'], INITIAL_MAPPING['word']),
    }
    for dataset in compound_dataset:
        # 3. Update each datasets type_to_idx mappings (for word and char types only)
        word_types = list(dataset.type_to_idx['word'])
        char_types = list(dataset.type_to_idx['char'])
        dataset.type_to_idx['word'] = Preprocessor.type_to_idx(word_types, type_to_idx['word'])
        dataset.type_to_idx['char'] = Preprocessor.type_to_idx(char_types, type_to_idx['char'])
        # 4. Re-compute the index sequences
        dataset.get_idx_seq()

    return compound_dataset


# TODO (John): This should be a `Dataset` method.
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


####################################################################################################
# Creating Validation Sets
####################################################################################################


def get_train_valid_indices(training_data, k_folds, shuffle=True):
    """Returns `k_folds` number of train/valid indices for inputs and targets in `training_data`.

    Returns a list of tuples, containing the train set and valid set indices for each of `k_folds`,
    which can be used to index `training_data['train']['x']` and `training_data['train']['y']`
    into train and valid partitions for k-fold cross-validation.

    Args:
        training_data (dict): A dict containing training data (inputs and targets) for a given
            dataset. Keyed by partition ('train', 'test', 'dev') and inputs ('x') and targets ('y')
            for each partition.
        k_folds (int): Number of folds to generate indices for.
        shuffle (bool): Optional, True if the data should be shuffled before splitting.
            Defaults to True.

    Returns:
        A list of tuples, where the ith tuple contains the train set and valid set indices for
        the ith of `k_folds` folds to be used for cross-validation.
    """
    kf = KFold(n_splits=k_folds, shuffle=shuffle, random_state=RANDOM_STATE)

    (X, _), y = training_data['train']['x'], training_data['train']['y']
    train_valid_indices = [(ti, vi) for ti, vi in kf.split(X, y)]

    return train_valid_indices


def get_data_partitions(training_data, train_valid_indices):
    """Splits `training_data` into `len(train_valid_indices)` number of folds for cross-validation.

    Returns a list of dictionaries, of length `len(train_valid_indices)` containing copies of
    `training_data` which as been partitioned for k-fold cross-validation.

    Args:
        training_data (dict): A dict containing training data (inputs and targets) for a given
            dataset. Keyed by partition ('train', 'test', 'dev') and inputs ('x') and targets ('y')
            for each partition.
        train_valid_indices (list): A list of two-tuples, where train_valid_indices[k].
            is a tuple containing the train and valid indices (in that order) for the kth fold.

    Returns:
        Returns `training_data_cv`, which is a list containing dictionaries, each storing one fold
        for cross-validation, created by splitting `training_data['train']`.
    """
    training_data_cv = []

    # Loop over folds
    for _, (train_indices, valid_indices) in enumerate(train_valid_indices):
        # Don't modify original dict
        training_data_cv.append(copy.deepcopy(training_data))

        x_train_word, x_train_char = training_data['train']['x']
        y_train = training_data['train']['y']

        x_train_word, x_valid_word = x_train_word[train_indices], x_train_word[valid_indices]
        x_train_char, x_valid_char = x_train_char[train_indices], x_train_char[valid_indices]
        y_train, y_valid = y_train[train_indices], y_train[valid_indices]

        training_data_cv[-1]['train']['x'] = [x_train_word, x_train_char]
        training_data_cv[-1]['train']['y'] = y_train

        training_data_cv[-1]['valid'] = {'x': [x_valid_word, x_valid_char], 'y': y_valid}

        # TODO (John): This is a hotfix to support RC
        if training_data['train']['orig_to_tok_map'] is not None:
            training_data_cv[-1]['train']['orig_to_tok_map'] = \
                training_data['train']['orig_to_tok_map'][train_indices]
            training_data_cv[-1]['valid']['orig_to_tok_map'] = \
                training_data['train']['orig_to_tok_map'][valid_indices]

        if training_data['train']['rel_labels'] is not None:
            training_data_cv[-1]['train']['rel_labels'] = \
                [training_data['train']['rel_labels'][k] for k in train_indices]
            training_data_cv[-1]['valid']['rel_labels'] = \
                [training_data['train']['rel_labels'][k] for k in valid_indices]

    return training_data_cv


def get_valid_data(training_data, validation_split=0.10):
    """Splits `training_data` into train and validation partitions.

    Generates a new training set of size 1 - `validation_split` and a validation set of size
    `validation_split` from the original training data given in `training_data`. Expects
    `training_data` to be a dictionary with the keyed by partition ('train', 'test', 'dev')
    containing the input examples and targets at keys 'x' and 'y' respectively, for each partition.

    Args:
        training_data (dict): A dict containing training data (inputs and targets) for a given
            dataset. Keyed by partition ('train', 'test', 'dev') and inputs ('x') and targets ('y')
            for each partition.
        validation_split (float): Percentage of the training examples in `training_data` to use for
            the new validation set.
        shuffle (bool): Optional, True if the data should be shuffled before splitting.
            Defaults to True.

    Returns:
        `training_data`, containing new train and validation sets at `training_data['train']['x']`
        / `training_data['train']['y']` and `training_data['valid']['x']` /
        `training_data['valid']['y']` respectively.
    """
    training_data_split = copy.deepcopy(training_data)  # don't modify original dict
    rs = ShuffleSplit(n_splits=1, test_size=validation_split, random_state=RANDOM_STATE)

    X, _ = training_data_split['train']['x']
    train_index, valid_index = next(rs.split(X))

    x_train_word, x_train_char = training_data_split['train']['x']
    y_train = training_data_split['train']['y']

    x_train_word, x_valid_word = x_train_word[train_index], x_train_word[valid_index]
    x_train_char, x_valid_char = x_train_char[train_index], x_train_char[valid_index]
    y_train, y_valid = y_train[train_index], y_train[valid_index]

    training_data_split['train']['x'] = x_train_word, x_train_char
    training_data_split['train']['y'] = y_train
    training_data_split['valid'] = {'x': (x_valid_word, x_valid_char), 'y': y_valid}

    # TODO (John): This is a hotfix to support RC
    if training_data_split['train']['orig_to_tok_map'] is not None:
        train_orig_to_tok_map = training_data_split['train']['orig_to_tok_map'][train_index]
        valid_orig_to_tok_map = training_data_split['train']['orig_to_tok_map'][valid_index]
        training_data_split['train']['orig_to_tok_map'] = train_orig_to_tok_map
        training_data_split['valid']['orig_to_tok_map'] = valid_orig_to_tok_map
    if training_data_split['train']['rel_labels'] is not None:
        train_rel_labels = [training_data_split['train']['rel_labels'][k] for k in train_index]
        valid_rel_labels = [training_data_split['train']['rel_labels'][k] for k in valid_index]
        training_data_split['train']['rel_labels'] = train_rel_labels
        training_data_split['valid']['rel_labels'] = valid_rel_labels

    return training_data_split


def get_cv_data(training_data, k_folds):
    """Splits training data (`training_data`) into `k_folds` number of train/valid partitions.

    Generates a train/valid split for `k_folds` number of cross-validation folds from the data
    given in `training_data`. Returns `partitioned_data`, a list containing dictionaries, where
    `partitioned_data[k]` contains the ith fold of `config.k_folds` folds for cross-validation.

    Args:
        training_data (dict): A dict containing training data (inputs and targets) for a given
            dataset. Keyed by partition ('train', 'test', 'dev') and inputs ('x') and targets ('y')
            for each partition.
        k_folds (int): Number of folds to split the data into.

    Returns:
        `partitioned_data`, a list containing dictionaries, where `partitioned_data[k]` contains the
        kth fold of `config.k_folds` folds for cross-validation.
    """
    train_valid_indices = get_train_valid_indices(training_data, k_folds)
    partitioned_data = get_data_partitions(training_data, train_valid_indices)

    return partitioned_data


def get_validation_set(config, training_data):
    """A helper function which collects a validation set (if none is provided).

    If there is no validation data provided (`training_data['valid'] is None`), then a validation
    set is created using one of two strategies. If `config.k_folds`, data is partitioned into
    `config.k_folds` for cross-validation. Otherwise, if `config.validation_split`, then
    `config.validation_split` percent of training examples are held out as a validation set from the
    train set if.

    Args:
        config (Config): A Config object which contains a set of harmonized arguments provided in
            a *.ini file and, optionally, from the command line.
        training_data (dict): A dict containing training data (inputs and targets) for a given
            dataset. Keyed by partition ('train', 'test', 'dev') and inputs ('x') and targets ('y')
            for each partition.

    Returns:
        If `training_data['valid'] is None`:
            A copy of `training_data`, unmodified.
        If `training_data['valid'] is not None and config.k_folds`:
            Returns `training_data_with_validation_set`, which is a list containing dictionaries,
            each storing one of `config.k_folds` folds for cross-validation, created by splitting
            `training_data['train']`.
        If `training_data['valid'] is not None and not `config.k_folds` and
            `config.validation_split`:
            Returns `training_data_with_validation_set`, which is a dictionary containing
            a copy of `training_data`, containing an additional validation set, created from
            `config.validation_split` proportion of examples in `training_data['train']`.
    """
    training_data_with_validation_set = training_data

    if training_data['valid'] is None:
        if config.k_folds:
            start = time.time()
            print(f'Creating {config.k_folds}-folds for k-fold cross-validation...',
                  end=' ', flush=True)

            training_data_with_validation_set = get_cv_data(training_data, config.k_folds)

            LOGGER.info(f'Partitioned data into {config.k_folds} folds for cross-validation')
            print(f'Done ({time.time() - start:.2f} seconds).')

        elif config.validation_split:
            start = time.time()
            print((f'Creating a validation set using a random {config.validation_split:.2%} of the'
                  ' train set...'), end=' ', flush=True)

            training_data_with_validation_set = \
                get_valid_data(training_data, config.validation_split)

            LOGGER.info((f'Used {config.validation_split:.2%} of the train set at random for'
                         ' validation'))
            print(f'Done ({time.time() - start:.2f} seconds).')

    return training_data_with_validation_set
