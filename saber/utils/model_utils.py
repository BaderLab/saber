"""A collection of model-related helper/utility functions.
"""
import logging
import os
from time import strftime

import torch
from google_drive_downloader import GoogleDriveDownloader as gdd

from .. import constants
from ..metrics import Metrics
from .generic_utils import extract_directory
from .generic_utils import make_dir

LOGGER = logging.getLogger(__name__)


####################################################################################################
# I/O
####################################################################################################


def prepare_output_directory(config):
    """Returns a list of output directories for each dataset folder in `config.dataset_folder`.

    For each dataset folder in `config.dataset_folder`, creates a directory (or directories when
    `len(config.dataset_folder) > 1`) under `config.output_folder` for saving the results of a
    training session, which includes performance evaluations (`evaluation.json`) and a copy of the
    config file used (`config.ini`). The directories are time stamped as `mmdd_HHMMSS`.

    The following directory structure is created:

    .
    ├── config.output_folder
    |   └── <first_dataset_name_second_dataset_name_nth_dataset_name>
    |       └── <first_dataset_name>
    |           └── mmdd_HHMMSS
    |       └── <second_dataset_name>
    |           └── mmdd_HHMMSS
    |       └── <nth_dataset_name>
    |           └── mmdd_HHMMSS

    If `len(config.dataset_folder) == 1`, only one directory is created under
    `config.output_folder`.

    Args:
        config (Config): A Config object which contains a set of harmonized arguments provided in
            a *.ini file and, optionally, from the command line.

    Returns:
        A list of directory paths to the time stamped subdirectories (`mmdd_HHMMSS`) which were
        created under `config.dataset_folder`, one for each dataset in `config.dataset_folder`.
    """
    output_dirs = []
    output_folder = config.output_folder
    # If multiple datasets, create additional directory to house all output directories
    if len(config.dataset_folder) > 1:
        dataset_names = '_'.join([os.path.basename(ds) for ds in config.dataset_folder])
        output_folder = os.path.join(output_folder, dataset_names)
    make_dir(output_folder)

    for dataset in config.dataset_folder:
        # Ceate a subdirectory for each datasets name
        dataset_dir = os.path.join(output_folder, os.path.basename(dataset))

        # Timestamp each datasets training output
        train_session_dir = strftime(r"%m%d_%H%M%S")
        dataset_train_session_dir = os.path.join(dataset_dir, train_session_dir)
        output_dirs.append(dataset_train_session_dir)
        make_dir(dataset_train_session_dir)

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


####################################################################################################
# Callbacks
####################################################################################################


def setup_metrics_callback(config, model, datasets, training_data, output_dirs):
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
    for i, (train_data, dataset, output) in enumerate(zip(training_data, datasets, output_dirs)):
        metric = Metrics(config=config,
                         model_=model,
                         training_data=train_data,
                         idx_to_tag=dataset.idx_to_tag,
                         output_dir=output,
                         model_idx=i)
        metrics.append(metric)

    return metrics


####################################################################################################
# Saving/loading
####################################################################################################


def download_model_from_gdrive(pretrained_model, extract=True):
    """Downloads a pre-trained Saber model from Google Drive.

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

    # Download model from Google Drive, will skip if already exists
    gdd.download_file_from_google_drive(file_id, dest_path=f'{dest_path}.tar.bz2', showsize=True)

    LOGGER.info('Loaded pre-trained model %s from Google Drive', pretrained_model)

    if extract:
        extract_directory(dest_path)
        return dest_path

    return f'{dest_path}.tar.bz2'


####################################################################################################
# PyTorch Helper Functions
####################################################################################################


def get_device(model=None):
    """Places `model` on CUDA device if available, returns PyTorch device, number of available GPUs.

    Returns a PyTorch device and number of available GPUs. If `model` is provided, and a CUDA device
    is available, the model is placed on the CUDA device. If multiple GPUs are available, the model
    is parallized with `torch.nn.DataParallel(model)`.

    Args:
        (Torch.nn.Module): A PyTorch model. If CUDA device is available this function will place the
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
