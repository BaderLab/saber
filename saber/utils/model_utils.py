"""A collection of model-related helper/utility functions.
"""
import os
from time import strftime

from keras.callbacks import ModelCheckpoint, TensorBoard

from ..metrics import Metrics
from .generic_utils import make_dir


def prepare_output_directory(config):
    """Creates an output directory under `output_folder` for each dataset in `dataset_folder`.

    Creates the following directory structure:
    .
    ├── output_folder
    |   └── <first_dataset_name_second_dataset_name_nth_dataset_name>
    |       └── <first_dataset_name>
    |           └── train_session_<month>_<day>_<hr>_<min>_<sec>
    |       └── <second_dataset_name>
    |           └── train_session_<month>_<day>_<hr>_<min>_<sec>
    |       └── <nth_dataset_name>
    |           └── train_session_<month>_<day>_<hr>_<min>_<sec>

    In the case of only a single dataset,
    <first_dataset_name_second_dataset_name_nth_dataset_name> and <first_dataset_name> are
    collapsed into a single directory. Saves a copy of the config file used to train the model
    (`config`) to the top level of this directory.

    Args:
        config (Config): A Config object which contains a set of harmonzied arguments provided in
            a *.ini file and, optionally, from the command line.

    Returns:
        a list of directory paths to the subdirectories
        train_session_<month>_<day>_<hr>_<min>_<sec>, one for each dataset in `dataset_folder`.
    """
    output_dirs = []
    output_folder = config.output_folder
    # if multiple datasets, create additional directory to house all output directories
    if len(config.dataset_folder) > 1:
        dataset_names = '_'.join([os.path.basename(ds) for ds in config.dataset_folder])
        output_folder = os.path.join(output_folder, dataset_names)

    make_dir(output_folder)

    for ds in config.dataset_folder:
        # create a subdirectory for each datasets name
        dataset_dir = os.path.join(output_folder, ds)
        # create a subdirectory for each train session
        train_session_dir = strftime("train_session_%a_%b_%d_%I_%M_%S").lower()
        dataset_train_session_dir = os.path.join(dataset_dir, train_session_dir)
        output_dirs.append(dataset_train_session_dir)
        make_dir(dataset_train_session_dir)

        # copy config file to top level directory
        config.save(dataset_train_session_dir)

    return output_dirs

def setup_checkpoint_callback(output_dir):
    """Sets up per epoch model checkpointing.

    Sets up model checkpointing by creating a Keras CallBack for each output directory in
    `output_dir` (correspondings to individual datasets).

    Args:
        output_dir (lst): A list of output directories, one for each dataset.

    Returns:
        checkpointer: A Keras CallBack object for per epoch model checkpointing.
    """
    checkpointers = []
    for dir_ in output_dir:
        metric_filepath = os.path.join(dir_, 'model_weights_best.hdf5')
        checkpointer = ModelCheckpoint(filepath=metric_filepath,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=True)
        checkpointers.append(checkpointer)

    return checkpointers

def setup_tensorboard_callback(output_dir):
    """Setup logs for use with TensorBoard.

    This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of
    your training and test metrics, as well as activation histograms for the different layers in
    your model. Logs are saved as `tensorboard_logs` at the top level of each directory in
    `output_dir`.

    Args:
        output_dir (lst): A list of output directories, one for each dataset.

    Returns:
        A list of Keras CallBack object for logging TensorBoard visulizations.

    Example:
        >>> tensorboard --logdir=/path_to_tensorboard_logs
    """
    tensorboards = []
    for dir_ in output_dir:
        tensorboard_dir = os.path.join(dir_, 'tensorboard_logs')
        tensorboards.append(TensorBoard(log_dir=tensorboard_dir))

    return tensorboards

def setup_metrics_callback(config, datasets, training_data, output_dir, fold=None):
    """Creates Keras Metrics Callback objects, one for each dataset in `datasets`.

    Args:
        config (Config): Contains a set of harmonzied arguments provided in a *.ini file and,
            optionally, from the command line.
        datasets (list): A list of Dataset objects.
        training_data (dict): A dictionary containing training data (inputs and targets).
        output_dir (list): List of directories to save model output to, one for each model.
        fold (int): The current fold in k-fold cross-validation.

    Returns:
        A list of Metric objects, one for each dataset in `datasets`.
    """
    metrics = []
    for i, ds in enumerate(datasets):
        metric = Metrics(config=config,
                         training_data=training_data[i],
                         index_map=ds.idx_to_tag,
                         output_dir=output_dir[i],
                         fold=fold)
        metrics.append(metric)

    return metrics

def setup_callbacks(config, output_dir):
    """Returns a list of Keras Callback objects to use during training.

    Args:
        config (Config): A Config object which contains a set of harmonzied arguments provided in
            a *.ini file and, optionally, from the command line.
        output_dir (list): A list of filepaths, one for each dataset in `self.datasets`.

    Returns:
        A list of Keras Callback objects to use during training.
    """
    callbacks = []
    # model checkpointing
    callbacks.append(setup_checkpoint_callback(output_dir))
    # tensorboard
    if config.tensorboard:
        callbacks.append(setup_tensorboard_callback(output_dir))

    return callbacks

def precision_recall_f1_support(true_positives, false_positives, false_negatives):
    """Returns the precision, recall, F1 and support from TP, FP and FN counts.

    Returns a four-tuple containing the precision, recall, F1-score and support
    For the given true_positive (TP), false_positive (FP) and
    false_negative (FN) counts.

    Args:
        true_positives (int): Number of true-positives predicted by classifier.
        false_positives (int): Number of false-positives predicted by classifier.
        false_negatives (int): Number of false-negatives predicted by classifier.

    Returns:
        Four-tuple containing (precision, recall, f1, support).
    """
    precision = true_positives / (true_positives + false_positives) if true_positives > 0 else 0.
    recall = true_positives / (true_positives + false_negatives) if true_positives > 0 else 0.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.
    support = true_positives + false_negatives

    return precision, recall, f1, support
