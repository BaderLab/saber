import os
from time import strftime

from keras import optimizers
from keras.callbacks import ModelCheckpoint

from utils_generic import make_dir

"""
A collection of model helper/utility functions.
"""

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
    if verbose: model.summary()

def create_train_session_dir(dataset_folder, output_folder):
    """Creates an output directory for each dataset in self.ds

    Creates the following directory structure:
    .
    ├── output_folder
    |   └── <first_dataset_name_second_dataset_name>
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
        a list directory paths to the subdirectories
        train_session_<month>_<day>_<hr>:<min>, one for each dataset in
        self.ds
    """
    # acc
    ts_output_dir = []

    # get a list of the dataset(s) name(s)
    ds_names = [os.path.basename(os.path.normpath(x)) for x in dataset_folder]

    if len(ds_names) > 1:
        # create a directory path composed of all dataset names
        top_level_ds_dir = os.path.join(output_folder, '_'.join(ds_names))
        # create list of subdirectories, one for each dataset
        ds_dirs = [os.path.join(top_level_ds_dir, ds) for ds in ds_names]
    else:
        ds_dirs = ds_names

    for dir in ds_dirs:
        # create a subdirectory for each datasets name
        ds_dir = os.path.join(output_folder, dir)
        # create a subdirectory for each train session
        ts_dir = strftime("train_session_%a_%b_%d_%I:%M").lower()
        # create the full directory path
        ds_ts_dir = os.path.join(ds_dir, ts_dir)
        ts_output_dir.append(ds_ts_dir)
        make_dir(ds_ts_dir)

    return ts_output_dir

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

    for dir in output_dir:
        # set up model checkpointing
        metric_filepath = os.path.join(dir, 'epoch_{epoch:02d}_checkpoint.hdf5')
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
