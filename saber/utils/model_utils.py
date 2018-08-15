"""A collection of model helper/utility functions.
"""
import os
from time import strftime

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import KFold

from ..metrics import Metrics
from .generic_utils import make_dir

# TODO (johngiorgi) add verbosity parameter for printing model summary

def compile_model(model, loss_function, optimizer, lr=0.01, decay=0.0, clipnorm=0.0):
    """Compiles a model specified with Keras.

    See https://keras.io/optimizers/ for more info on each optmizer.

    Args:
        model: Keras model object to compile
        loss_function: Keras loss_function object to compile model with
        optimizer (str): the optimizer to use during training
        lr (float): learning rate to use during training
        decay (float): per epoch decay rate
        clipnorm (float): gradient normalization threshold
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

def prepare_output_directory(dataset_folder, output_folder, config=None):
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
    collapsed into a single directory. Also saves a copy of the config file
    used to train the model to the top level of this directory.

    Args:
        dataset_folder (str): a list of directory paths to CoNLL formatted
            datasets
        output_folder (str): the top-level output folder
        config_filepath (str): optional, if not None, the config file used
            to train a model is copied to the top-level of the output directory.

    Returns:
        a list of directory paths to the subdirectories
        train_session_<month>_<day>_<hr>_<min>, one for each dataset in
        dataset_folder
    """
    # acc
    train_sess_output_dirnames = []

    # create a directory path composed of all dataset names
    ds_names = [os.path.basename(x) for x in dataset_folder]
    top_level_dir = os.path.join(output_folder, '_'.join(ds_names))
    make_dir(top_level_dir)

    # create list of subdirectories, one for each dataset
    if len(ds_names) > 1:
        ds_dirs = [os.path.join(top_level_dir, ds) for ds in ds_names]
    else:
        ds_dirs = ds_names

    for dirname in ds_dirs:
        # create a subdirectory for each datasets name
        ds_dirname = os.path.join(output_folder, dirname)

        # create a subdirectory for each train session
        ts_dirname = strftime("train_session_%a_%b_%d_%I_%M").lower()

        # create the full directory path
        ds_ts_dirname = os.path.join(ds_dirname, ts_dirname)
        train_sess_output_dirnames.append(ds_ts_dirname)
        make_dir(ds_ts_dirname)

        # copy config file to top level directory
        if config is not None:
            config.save(ds_ts_dirname)

    return train_sess_output_dirnames

def setup_checkpoint_callback(output_dir):
    """Sets up per epoch model checkpointing.

    Sets up model checkpointing by creating a Keras CallBack for each
    output directory in output_dir (correspondings to individual datasets).

    Args:
        output_dir (lst): a list of output directories, one for each dataset

    Returns:
        checkpointer: a Keras CallBack object for per epoch model checkpointing.
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

def setup_tensorboard_callback(output_dir):
    """Setup logs for use with TensorBoard.

    This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of
    your training and test metrics, as well as activation histograms for the different layers in
    your model. Logs are saved as `tensorboard_logs` at the top level of each directory in
    `output_dir`.

    Args:
        output_dir (lst): a list of output directories, one for each dataset

    Returns:
        a list of Keras CallBack object for logging TensorBoard visulizations.

    Example:
        >>> tensorboard --logdir=/path_to_tensorboard_logs
    """
    tensorboards = []
    for dir_ in output_dir:
        tensorboard_dir = os.path.join(dir_, 'tensorboard_logs')
        tensorboards.append(TensorBoard(log_dir=tensorboard_dir))

    return tensorboards

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
    precision = true_positives / (true_positives + false_positives) if true_positives > 0 else 0.
    recall = true_positives / (true_positives + false_negatives) if true_positives > 0 else 0.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.
    support = true_positives + false_negatives

    return precision, recall, f1, support

def get_train_valid_indices(training_data, k_folds):
    """Get `k_folds` number of sets of train/valid indices for all datasets in `training_data`.

    For all datatsets in `training_data`, gets `k_folds` number of train and valid indices. Returns
    a list of list of two-tuples, where the outer list is of length len(training_data) and the inner
    list is of length `k_folds` and contains two-tuples corresponding to train and valid indicies
    respectively. For example, the train indicies for the ith dataset and jth fold would be
    train_valid_indices[i][j][0].

    Args:
        training_data (dict): a list containing i dictionaries, one for each datasets, which
            in turn contain the keys 'X_train', 'X_valid', 'y_train', 'y_valid'.
        k_folds (int): number of folds to compute indices for

    Returns:
        a list of list of two-tuples, where index [i][j] is a tuple containing the train and valid
        indicies (in that order) for the ith dataset and jth k-fold.
    """
    # global acc
    train_valid_indices = []
    # Sklearn KFold object
    kf = KFold(n_splits=k_folds, random_state=42)

    for i in training_data:
        X, _ = training_data[i]['X_train']
        train_valid_indices.append([(ti, vi) for ti, vi in kf.split(X)])

    return train_valid_indices

def get_data_partitions(training_data, train_valid_indices, fold):
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
        a list of six-tuples containing train and valid data for all datasets.
    """
    # TODO: this seems like a sub-par solution
    # acc, p = partition, s = split
    partitioned_data = {p: {s: None for s in training_data[p]} for p in training_data}

    for i in training_data:
        # train_valid_indices[i][fold] is a two-tuple, where index 0 contains
        # the train indices and index 1 the valid indicies
        train_indices, valid_indices = train_valid_indices[i][fold]
        # get training data for dataset i
        X_word, X_char = training_data[i]['X_train']
        # get labels for dataset i
        y = training_data[i]['y_train']

        # create and accumulate train/valid partitions
        X_word_train, X_word_valid = X_word[train_indices], X_word[valid_indices]
        X_char_train, X_char_valid = X_char[train_indices], X_char[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]

        partitioned_data[i]['X_train'] = [X_word_train, X_char_train]
        partitioned_data[i]['X_valid'] = [X_word_valid, X_char_valid]
        partitioned_data[i]['y_train'] = y_train
        partitioned_data[i]['y_valid'] = y_valid

    return partitioned_data

def get_metrics(datasets, training_data, output_dir, criteria='exact', fold=None):
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
    metrics = []

    for i, ds in enumerate(datasets):
        metrics_ = Metrics(
            training_data=training_data[i],
            idx_to_tag=ds.idx_to_tag,
            output_dir=output_dir[i],
            criteria=criteria,
            fold=fold)
        metrics.append(metrics_)

    return metrics

def idx_seq_to_categorical(idx_seq, num_classes=None):
    """One-hot encodes a given class vector.

    Converts a class matrix of integers, `idx_seq`, of shape (num examples, sequence length) to a
    one-hot encoded matrix of shape (num_examples, sequence length, num_classes).

    Args:
        idx_seq: class matrix of integers of shape (num examples, sequence length), representing a
            sequence of tags

    Returns:
        numpy array, one-hot encoded matrix representation of `idx_seq` of shape
            (num examples, sequence length, num_classes)
    """
    # convert to one-hot encoding
    one_hots = [to_categorical(s, num_classes) for s in idx_seq]
    one_hots = np.array(one_hots)

    return one_hots
