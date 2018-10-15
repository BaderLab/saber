"""A collection of generic helper/utility functions.
"""
import errno
import logging
import os
import shutil

from setuptools.archive_util import unpack_archive

from .. import constants

log = logging.getLogger(__name__)

# https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist#273227
def make_dir(directory):
    """Creates a directory at `directory` if it does not already exist.
    """
    try:
        os.makedirs(directory)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

def clean_path(filepath):
    """Returns normalized and absolutized `filepath`.
    """
    filepath = filepath.strip() if isinstance(filepath, str) else filepath
    return os.path.abspath(os.path.normpath(filepath))

def extract_directory(directory):
    """Extracts bz2 compressed directory at `directory` if directory is compressed.
    """
    if not os.path.isdir(directory):
        head, _ = os.path.split(os.path.abspath(directory))

        print('\nunzipping... ', end='', flush=True)
        unpack_archive(directory + '.tar.bz2', extract_dir=head)

def compress_directory(directory):
    """Compresses a given directory using bz2 compression.

    Raises:
        ValueError: if no directory at `directory` exists or if `directory`.tar.bz2 already exists.
    """
    # clean/normalize directory
    directory = os.path.abspath(os.path.normcase(os.path.normpath(directory)))

    # raise ValueError if directory.tar.bz2 already exists or if directory not valid
    output_filepath = '{}.tar.bz2'.format(directory)
    if os.path.exists(output_filepath):
        err_msg = "{} already exists.".format(output_filepath)
        log.error('ValueError %s', err_msg)
        raise ValueError(err_msg)
    if not os.path.exists(directory):
        err_msg = "File or directory at 'directory' does not exist."
        log.error('ValueError %s', err_msg)
        raise ValueError(err_msg)

    # create bz2 compressed directory, remove uncompressed directory
    root_dir = os.path.abspath(''.join(os.path.split(directory)[:-1]))
    base_dir = os.path.basename(directory)
    shutil.make_archive(base_name=directory, format='bztar', root_dir=root_dir, base_dir=base_dir)
    shutil.rmtree(directory)

def get_pretrained_model_dir(config):
    """Returns path to top-level directory to save a pretrained model.

    Returns a directory path to save a pretrained model based on `config.dataset_folder` and
    `config.output_folder`. The folder which contains the saved model is named from each dataset
    name in `config.dataset_folder` joined by an underscore. The full path is:
    `<config.output_folder>/<constants.PRETRAINED_MODEL_DIR>/<dataset_names>`

    Args:
        config (Config): Config object

    Returns:
        full path to save a pre-trained model based on 'config.dataset_folder' and
        'config.dataset_folder'
    """
    ds_names = '_'.join([os.path.basename(ds) for ds in config.dataset_folder])
    return os.path.join(config.output_folder, constants.PRETRAINED_MODEL_DIR, ds_names)
