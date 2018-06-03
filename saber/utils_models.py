"""A collection of model helper/utility functions.
"""
import os
from time import strftime

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

import metrics
from utils_generic import make_dir

# TODO (johngiorgi) add verbosity parameter for printing model summary

def compile_model(model,
                  loss_function,
                  optimizer,
                  lr=0.01,
                  decay=0.0,
                  clipnorm=None,
                  verbose=False):
    """Compiles a model specified with Keras.

    See https://keras.io/optimizers/ for more info on each optmizer.

    Args:
        model: Keras model object to compile
        loss_function: Keras loss_function object to compile model with
        optimizer (str): the optimizer to use during training
        lr (float): learning rate to use during training
        decay (float): per epoch decay rate
        clipnorm (float): gradient normalization threshold
        verbose (bool): if True, prints model summary after compilation
    """
    # The parameters of these optimizers can be freely tuned.
    if optimizer == 'sgd':
        optimizer_ = optimizers.SGD(lr=lr, decay=decay, clipnorm=clipnorm)
    elif optimizer == 'adam':
        optimizer_ = optimizers.Adam(lr=lr, decay=decay, clipnorm=clipnorm)
    elif optimizer == 'adamax':
        optimizer_ = optimizers.Adamax(lr=lr, decay=decay, clipnorm=clipnorm)
    # It is recommended to leave the parameters of this optimizer at their
    # default values (except the learning rate, which can be freely tuned).
    # This optimizer is usually a good choice for recurrent neural networks
    elif optimizer == 'rmrprop':
        optimizer_ = optimizers.RMSprop(lr=lr, clipnorm=clipnorm)
    # It is recommended to leave the parameters of these optimizers at their
    # default values.
    elif optimizer == 'adagrad':
        optimizer_ = optimizers.Adagrad(clipnorm=clipnorm)
    elif optimizer == 'adadelta':
        optimizer_ = optimizers.Adadelta(clipnorm=clipnorm)
    elif optimizer == 'nadam':
        optimizer_ = optimizers.Nadam(clipnorm=clipnorm)

    model.compile(optimizer=optimizer_, loss=loss_function)
    if verbose: print(model.summary())

def create_train_session_dir(dataset_folder, output_folder):
    """Creates an output directory for each dataset in dataset_folder

    Creates the following directory structure:
    .
    ├── output_folder
    |   └── <first_dataset_name_second_dataset_name_nth_dataset_name>
    |       └── <first_dataset_name>
    |           └── train_session_<month>_<day>_<hr>:<min>
    |       └── <second_dataset_name>
    |           └── train_session_<month>_<day>_<hr>:<min>
    |       └── <nth_dataset_name>
    |           └── train_session_<month>_<day>_<hr>:<min>

    In the case of only a single dataset,
    <first_dataset_name_second_dataset_name> and <first_dataset_name> are
    collapsed into a single directory.

    Returns:
        a list of directory paths to the subdirectories
        train_session_<month>_<day>_<hr>:<min>, one for each dataset in
        dataset_folder
    """
    # acc
    ts_output_dirnames = []

    # get a list of the dataset(s) name(s)
    ds_names = [os.path.basename(os.path.normpath(x)) for x in dataset_folder]

    if len(ds_names) > 1:
        # create a directory path composed of all dataset names
        top_level_ds_dir = os.path.join(output_folder, '_'.join(ds_names))
        # create list of subdirectories, one for each dataset
        ds_dirs = [os.path.join(top_level_ds_dir, ds) for ds in ds_names]
    else:
        ds_dirs = ds_names

    for dirname in ds_dirs:
        # create a subdirectory for each datasets name
        ds_dirname = os.path.join(output_folder, dirname)
        # create a subdirectory for each train session
        ts_dirname = strftime("train_session_%a_%b_%d_%I:%M").lower()
        # create the full directory path
        ds_ts_dirname = os.path.join(ds_dirname, ts_dirname)
        ts_output_dirnames.append(ds_ts_dirname)
        make_dir(ds_ts_dirname)

    return ts_output_dirnames

def setup_model_checkpointing(output_dir):
    """Sets up per epoch model checkpointing.

    Sets up model checkpointing by creating a Keras CallBack for each
    output directory in output_dir (correspondings to invidivual datasets).

    Returns:
        checkpointer: a Keras CallBack object for per epoch model
                      checkpointing.
    """
    # acc
    checkpointers = []

    for dir_ in output_dir:
        # set up model checkpointing
        metric_filepath = os.path.join(dir_, 'model_weights_best.hdf5')
        checkpointers.append(
            ModelCheckpoint(filepath=metric_filepath,
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True))

    return checkpointers

def precision_recall_f1_support(true_positives, false_positives, false_negatives):
    """Returns the precision, recall, F1 and support from TP, FP and FN counts.

    Returns a four-tuple containing the precision, recall, F1-score and support
    For the given true_positive (TP), false_positive (FP) and
    false_negative (FN) counts.

    Args:
        true_positives (int): number of true-positives predicted by classifier
        false_positives (int): number of false-positives predicted by classifier
        false_negatives (int): number of false-negatives predicted by classifier

    Returns:
        four-tuple containing (precision, recall, f1, support)
    """
    precision = true_positives / (true_positives + false_positives) \
        if true_positives > 0 else 0.
    recall = true_positives / (true_positives + false_negatives) \
        if true_positives > 0 else 0.
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.
    support = true_positives + false_negatives

    return precision, recall, f1, support

def get_train_valid_indices(datasets, k_folds):
    """Get train and valid indicies for all k-folds for all datasets.

    For all datatsets ds, gets k_folds number of train and valid indicies.
    Returns a list of list of two-tuples, where the outer list is of length
    len(ds) and the inner list is of length k_folds and contains two-tuples
    corresponding to train indicies and valid indicies respectively. The train
    indicies for the ith dataset and jth fold would be
    compound_train_valid_indices[i][j][0].

    Args:
        datasets (list): a list of Dataset objects
        k_folds (int): number of k folds to preform in cross-validation

    Returns:
        compound_train_valid_indices: a list of list of two-tuples, where
        compound_train_valid_indices[i][j] is a tuple containing the train
        and valid indicies (in that order) for the ith dataset and jth
        k-fold.
    """
    # acc
    compound_train_valid_indices = []
    # Sklearn KFold object
    kf = KFold(n_splits=k_folds, random_state=42)

    for ds in datasets:
        X = ds.train_word_idx_seq
        # acc
        dataset_train_valid_indices = []
        for train_idx, valid_idx in kf.split(X):
            dataset_train_valid_indices.append((train_idx, valid_idx))
        compound_train_valid_indices.append(dataset_train_valid_indices)

    return compound_train_valid_indices

def get_data_partitions(datasets, train_valid_indices, fold):
    """Get train and valid partitions for all k-folds for all datasets.

    For all datasets, gets the train and valid partitions for
    all k folds. Returns a list of six-tuples:

    (X_train_word, X_valid_word, X_train_char, X_valid_char, y_train, y_valid)

    Where X represents the inputs, and y the labels. Inputs include
    sequences of words (X_word), and sequences of characters (X_char)

    Args:
        datasets (list): a list of Dataset objects
        train_valid_indices (list): a list of list of two-tuples, where
            train_valid_indices[i][j] is a tuple containing the train and valid
            indicies (in that order) for the ith dataset and jth fold
        fold (int): the current fold in k-fold cross-validation


    Returns:
        six-tuple containing train and valid data for all datasets.
    """
    # acc
    data_partition = []

    for i, ds in enumerate(datasets):
        X_word = ds.train_word_idx_seq
        X_char = ds.train_char_idx_seq
        y = ds.train_tag_idx_seq
        # train_valid_indices[i][fold] is a two-tuple, where index
        # 0 contains the train indices and index 1 the valid
        # indicies
        X_word_train = X_word[train_valid_indices[i][fold][0]]
        X_word_valid = X_word[train_valid_indices[i][fold][-1]]
        X_char_train = X_char[train_valid_indices[i][fold][0]]
        X_char_valid = X_char[train_valid_indices[i][fold][-1]]
        y_train = y[train_valid_indices[i][fold][0]]
        y_valid = y[train_valid_indices[i][fold][-1]]

        data_partition.append((X_word_train,
                               X_word_valid,
                               X_char_train,
                               X_char_valid,
                               y_train,
                               y_valid))

    return data_partition

def get_metrics(datasets, data_partitions, output_dir, fold=1):
    """Creates Keras Metrics Callback objects, one for each dataset.

    Args:
        datasets (list): a list of Dataset objects
        data_paritions (tuple): six-tuple containing train/valid data for all ds
        output_dir (str): path to directory to save metric output files
        fold (int): the current fold in k-fold cross-val

    Returns:
        a list of Metric objects, one for each dataset in datasets.
    """
    # acc
    metrics_acc = []

    for i, ds in enumerate(datasets):
        # data_partitions[i] is a four-tuple where index 0 contains X_train
        # data partition, index 1 X_valid data partition, ..., for dataset i
        X_word_train = data_partitions[i][0]
        X_word_valid = data_partitions[i][1]
        X_char_train = data_partitions[i][2]
        X_char_valid = data_partitions[i][3]
        y_train = data_partitions[i][4]
        y_valid = data_partitions[i][5]

        metrics_acc.append(metrics.Metrics([X_word_train, X_char_train],
                                           [X_word_valid, X_char_valid],
                                           y_train, y_valid,
                                           idx_to_tag_type = ds.idx_to_tag_type,
                                           output_dir=output_dir[i],
                                           fold=fold))

    return metrics_acc
