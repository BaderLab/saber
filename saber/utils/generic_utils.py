"""A collection of generic helper/utility functions.
"""
import errno
import logging
import os
import shutil

from setuptools.archive_util import unpack_archive

LOGGER = logging.getLogger(__name__)

def is_consecutive(lst):
    """Returns True if `lst` contains all numbers from 0 to `len(lst)` with no duplicates.
    """
    return sorted(lst) == list(range(len(lst)))

def reverse_dict(mapping):
    """Returns a dictionary composed of the reverse v, k pairs of a dictionary `mapping`.
    """
    return {v: k for k, v in mapping.items()}

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

        print('Unzipping...', end=' ', flush=True)
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
        LOGGER.error('ValueError %s', err_msg)
        raise ValueError(err_msg)
    if not os.path.exists(directory):
        err_msg = "File or directory at 'directory' does not exist."
        LOGGER.error('ValueError %s', err_msg)
        raise ValueError(err_msg)

    # create bz2 compressed directory, remove uncompressed directory
    root_dir = os.path.abspath(''.join(os.path.split(directory)[:-1]))
    base_dir = os.path.basename(directory)
    shutil.make_archive(base_name=directory, format='bztar', root_dir=root_dir, base_dir=base_dir)
    shutil.rmtree(directory)
