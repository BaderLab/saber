"""A collection of model-related helper/utility functions.
"""
import logging
import os
from time import strftime

import numpy as np
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from .. import constants
from ..metrics import Metrics
from .generic_utils import extract_directory
from .generic_utils import make_dir

LOGGER = logging.getLogger(__name__)


# I/O

def prepare_output_directory(config):
    """Create output directories `config.output_folder/config.dataset_folder` for each dataset.

    Creates the following directory structure:
    .
    ├── config.output_folder
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
        config (Config): A Config object which contains a set of harmonized arguments provided in
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

    for dataset in config.dataset_folder:
        # create a subdirectory for each datasets name
        dataset_dir = os.path.join(output_folder, os.path.basename(dataset))
        # create a subdirectory for each train session
        train_session_dir = strftime("train_session_%a_%b_%d_%I_%M_%S").lower()
        dataset_train_session_dir = os.path.join(dataset_dir, train_session_dir)
        output_dirs.append(dataset_train_session_dir)
        make_dir(dataset_train_session_dir)

        # copy config file to top level directory
        config.save(dataset_train_session_dir)

    return output_dirs


def prepare_pretrained_model_dir(config):
    """Returns path to top-level directory to save a pre-trained model.

    Returns a directory path to save a pre-trained model based on `config.dataset_folder` and
    `config.output_folder`. The folder which contains the saved model is named from each dataset
    name in `config.dataset_folder` joined by an underscore:
    .
    ├── config.output_folder
    |   └── <constants.PRETRAINED_MODEL_DIR>
    |       └── <first_dataset_name_second_dataset_name_nth_dataset_name>

    config (Config): A Config object which contains a set of harmonized arguments provided in
        a *.ini file and, optionally, from the command line.

    Returns:
        Full path to save a pre-trained model based on `config.dataset_folder` and
        `config.dataset_folder`.
    """
    ds_names = '_'.join([os.path.basename(ds) for ds in config.dataset_folder])
    return os.path.join(config.output_folder, constants.PRETRAINED_MODEL_DIR, ds_names)

# Keras callbacks


def setup_checkpoint_callback(config, output_dir):
    """Sets up per epoch model checkpointing.

    Sets up model checkpointing by creating a Keras CallBack for each output directory in
    `output_dir` (corresponding to individual datasets).

    Args:
        output_dir (lst): A list of output directories, one for each dataset.

    Returns:
        checkpointer: A Keras CallBack object for per epoch model checkpointing.
    """
    checkpointers = []
    for dir_ in output_dir:
        # if only saving best weights, filepath needs to be the same so it gets overwritten
        if config.save_all_weights:
            filepath = os.path.join(dir_, 'weights_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.hdf5')
        else:
            filepath = os.path.join(dir_, constants.WEIGHTS_FILENAME)

        checkpointer = ModelCheckpoint(filepath=filepath,
                                       monitor='val_loss',
                                       save_best_only=(not config.save_all_weights),
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
        A list of Keras CallBack object for logging TensorBoard visualizations.

    Example:
        >>> tensorboard --logdir=/path_to_tensorboard_logs
    """
    tensorboards = []
    for dir_ in output_dir:
        tensorboard_dir = os.path.join(dir_, 'tensorboard_logs')
        tensorboards.append(TensorBoard(log_dir=tensorboard_dir))

    return tensorboards


def setup_metrics_callback(model, datasets, config, training_data, output_dir, fold=None):
    """Creates Keras Metrics Callback objects, one for each dataset in `datasets`.

    Args:
        model (BaseModel): Model to evaluate, subclass of BaseModel.
        datasets (list): A list of Dataset objects.
        config (Config): Contains a set of harmonzied arguments provided in a *.ini file and,
            optionally, from the command line.
        training_data (list): A list of dictionaries, where the ith element contains the data
            for the ith dataset indices and the dictionaries keys are dataset partitions
            ('x_train', 'y_train', 'x_valid', ...)
        output_dir (list): List of directories to save model output to, one for each model.
        fold (int): The current fold in k-fold cross-validation. Defaults to None.

    Returns:
        A list of Metric objects, one for each dataset in `datasets`.
    """
    metrics = []
    for i, dataset in enumerate(datasets):
        eval_data = training_data[i] if fold is None else training_data[i][fold]
        metric = Metrics(config=config,
                         model_=model,
                         training_data=eval_data,
                         idx_to_tag=dataset.idx_to_tag,
                         output_dir=output_dir[i],
                         model_idx=i,
                         fold=fold)
        metrics.append(metric)

    return metrics


def setup_callbacks(config, output_dir):
    """Returns a list of Keras Callback objects to use during training.

    Args:
        config (Config): A Config object which contains a set of harmonized arguments provided in
            a *.ini file and, optionally, from the command line.
        output_dir (list): A list of filepaths, one for each dataset in `self.datasets`.

    Returns:
        A list of Keras Callback objects to use during training.
    """
    callbacks = []
    # model checkpointing
    callbacks.append(setup_checkpoint_callback(config, output_dir))
    # tensorboard
    if config.tensorboard:
        callbacks.append(setup_tensorboard_callback(output_dir))

    return callbacks


# Evaluation metrics

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
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.
    support = true_positives + false_negatives

    return precision, recall, f1_score, support


def mask_labels(y_true, y_pred, label):
    """Masks all instances of `label` from `y_true` and `y_pred` where `y_true == label`.

    Masks (removes) all indices in `y_true` and `y_pred` where `y_true` is equal to
    `label`. This step is necessary for discarding the sequence pad labels (and predictions made on
    these sequence pad labels) from the gold labels and model predictions before performance metrics
    are computed. Note that the returned elements are lists.

    Args:
        y_true (array_like): 2D array containing gold labels.
        y_pred (array_like): 2D array containing predicted labels.
        label (int): The label, or index to mask from the sequences `y_true` and `y_pred`.

    Returns:
        A tuple of lists, containing `y_true` and `y_pred`, where all indices where `y_true` was
        equal to `label` have been removed.
    """
    mask = y_true == label

    y_true_masked = np.ma.masked_array(y_true, mask).tolist()
    y_pred_masked = np.ma.masked_array(y_pred, mask).tolist()

    y_true = [[y for y in sent if y is not None] for sent in y_true_masked]
    y_pred = [[y for y in sent if y is not None] for sent in y_pred_masked]

    return y_true, y_pred


# Saving/loading

def load_pretrained_model(config, datasets, model_filepath, weights_filepath=None, **kwargs):
    """Loads a pre-trained Keras model from its pre-trained weights and architecture files.

    Loads a pre-trained Keras model given by its pre-trained weights (`weights_filepath`) and
    architecture files (`model_filepath`). The type of model to load is specificed in
    `config.model_name`.

    Args:
        config (Config): config (Config): A Config object which contains a set of harmonized
            arguments provided in a *.ini file and, optionally, from the command line.
        datasets (Dataset): A list of Dataset objects.
        model_filepath (str): A filepath to the architecture of a pre-trained model. For PyTorch
            models, this contains everything we need to load the model. For Keras models, you
            must additionally supply the weights in `weights_filepath`.
        weights_filepath (str): A filepath to the weights of a pre-trained model. This is not
            required for PyTorch models. Defaults to None.

    Returns:
        A pre-trained model.
    """
    # import statements are here to prevent circular imports
    if config.model_name == 'bilstm-crf-ner':
        from ..models.bilstm_crf import BiLSTMCRF
        model = BiLSTMCRF(config, datasets)
        model.load(model_filepath, weights_filepath)
    elif config.model_name == 'bert-ner':
        from ..models.bert_for_ner import BertForNER
        model = BertForNER(config, datasets, kwargs['pretrained_model_name_or_path'])
        model.load(model_filepath)

    model.compile()

    return model


def download_model_from_gdrive(pretrained_model, extract=True):
    """Downloads a pre-trained model from Google Drive.

    Args:
        pretrained_model (str): The name of a pre-trained model. Must be in
            `constants.PRETRAINED_MODELS`.
        extract (bool): Optional, True if downloaded tar.bz2 file should be extracted. Defaults to
            True.

    Returns:
        The filepath of the pre-trained model. This will be an uncompressed directory if `extract`
        or a compressed directory otherwise.
    """
    file_id = constants.PRETRAINED_MODELS[pretrained_model]
    dest_path = os.path.join(constants.PRETRAINED_MODEL_DIR, pretrained_model)

    # download model from Google Drive, will skip if already exists
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=f'{dest_path}.tar.bz2',
                                        showsize=True)

    LOGGER.info('Loaded pre-trained model %s from Google Drive', pretrained_model)

    if extract:
        extract_directory(dest_path)
        return dest_path

    return f'{dest_path}.tar.bz2'


# Keras helper functions

def get_keras_optimizer(optimizer, lr=0.01, decay=0.0, clipnorm=0.0):
    """A Keras helper function that initializes and returns the optimizer `optimizer`.

    Args:
        optimizer (str): Name of a valid Keras optimizer.
        lr (float): Optional, learning rate of the optimizer. Defaults to 0.01.
        decay (float): Optional, decay rate of the optimizer. Defaults to 0.0.
        clipnorm (float): Optional, L2 norm to clip all gradients to. Defaults to 0.0.

    Returns:
        An initialized Keras optimizer with name `optimizer`.
    """
    # The parameters of these optimizers can be freely tuned.
    if optimizer == 'sgd':
        optimizer = optimizers.SGD(lr=lr, decay=decay, clipnorm=clipnorm)
    elif optimizer == 'adam':
        optimizer = optimizers.Adam(lr=lr, decay=decay, clipnorm=clipnorm)
    elif optimizer == 'adamax':
        optimizer = optimizers.Adamax(lr=lr, decay=decay, clipnorm=clipnorm)
    # It is recommended to leave the parameters of this optimizer at their
    # default values (except the learning rate, which can be freely tuned).
    # This optimizer is usually a good choice for recurrent neural networks
    elif optimizer == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=lr, clipnorm=clipnorm)
    # It is recommended to leave the parameters of these optimizers at their
    # default values.
    elif optimizer == 'adagrad':
        optimizer = optimizers.Adagrad(clipnorm=clipnorm)
    elif optimizer == 'adadelta':
        optimizer = optimizers.Adadelta(clipnorm=clipnorm)
    elif optimizer == 'nadam':
        optimizer = optimizers.Nadam(clipnorm=clipnorm)
    else:
        err_msg = (f'Expected `optimizer` to be a string representing a valid optimizer name in'
                   ' Keras. Got: {optimizer}')
        LOGGER.error('ValueError %s', err_msg)
        raise ValueError(err_msg)

    return optimizer


def freeze_output_layers(model, model_idx):
    """Freeze output layers of Keras model `model` besides layer at `model_idx`.

    Freezes all output layers of a Keras model (`model`) besides the `model_idx` numbered output
    layer.

    Args:
        model (keras.models.Model): Keras model to modify.
        model_idx (int): Index into the output layer to remain trainable (unfrozen).
    """
    n_outputs = len(model.output)
    n_layers = len(model.layers)

    for i, _ in enumerate(model.output):
        layer = model.get_layer(index=n_layers - n_outputs + i)
        if i == model_idx:
            layer.trainable = True
        else:
            layer.trainable = False


def get_targets(training_data, model_idx, fold=None):
    """Returns a tuple of train, valid targets modified for use with a multi-task model.

    Given `training_data`, a list of data for training a multi-task model, zeros out the train and
    valid targets for all datasets not corresponding to the current output layer being trained
    (`model_idx`).

    Args:
        training_data (list): A list of dicts containing the data (at key `x_partition`) and targets
            (at key `y_partition`) for each partition: 'train', 'valid' and 'test'.
        model_idx (int): Index to `training_data` which corresponds to the current task
            being trained (i.e. the current dataset and output layer).
        fold (int): Optional, integer corresponding to the current fold in k-fold cross validation.
            Should be None if not training using cross-validation. Defaults to None.

    Returns:
        Tuple of train, valid targets modified for use with a multi-task model.
    """
    if fold is None:
        current_train_target = training_data[model_idx]['y_train']
        current_valid_target = training_data[model_idx]['y_valid']
    else:
        current_train_target = training_data[model_idx][fold]['y_train']
        current_valid_target = training_data[model_idx][fold]['y_valid']

    train_targets = [np.zeros_like(current_train_target) for _, _ in enumerate(training_data)]
    valid_targets = [np.zeros_like(current_valid_target) for _, _ in enumerate(training_data)]

    train_targets[model_idx] = current_train_target
    valid_targets[model_idx] = current_valid_target

    return train_targets, valid_targets


# PyTorch helper functions

def get_device(model=None):
    """Places `model` on CUDA device if available, returns PyTorch device, number of available GPUs.

    Returns a PyTorch device and number of available GPUs. If `model` is provided, and a CUDA device
    is available, the model is placed on the CUDA device. If multiple GPUs are available, the model
    is parallized with `torch.nn.DataParallel(model)`.

    Args:
        (Torch.nn.Module) PyTorch model, if CUDA device is available this function will place the
        model on the CUDA device with `model.to(device)`. If multiple CUDA devices are available,
        the model is parallized with `torch.nn.DataParallel(model)`.

    Returns:
        A two-tuple containing a PyTorch device ('cpu' or 'cuda'), and number of CUDA devices
        available.
    """
    n_gpu = 0

    # use a GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        n_gpu = torch.cuda.device_count()
        # if model is provided, we place it on the GPU and parallize it (if possible)
        if model:
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
        model_names = ', '.join([torch.cuda.get_device_name(i) for i in range(n_gpu)])
        print('Using CUDA device(s) with name(s): {}.'.format(model_names))
    else:
        device = torch.device('cpu')
        print('No GPU available. Using CPU.')

    return device, n_gpu
