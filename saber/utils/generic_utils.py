"""A collection of generic helper/utility functions.
"""
import os
import errno
import codecs
import shutil
from setuptools.archive_util import unpack_archive

from .. import constants

# https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist#273227
def make_dir(directory_filepath):
    """Creates a directory (directory_filepath) if it does not exist.

    Args:
        directory_filepath: filepath of directory to create
    """
    # create output directory if it does not exist
    try:
        os.makedirs(directory_filepath)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

def decompress_model(filepath):
    """Decompresses a bz2 compressed Saber model.

    If filepath is not a directory, decompresses the identically named bz2 Saber
    model at filepath.

    Args:
        filepath (str): path to a pre-trained Saber model (zipped or unzipped)
    """
    if not os.path.isdir(filepath):
        head, _ = os.path.split(os.path.abspath(filepath))

        print('[INFO] Unzipping pretrained model... '.format(), end='', flush=True)
        unpack_archive(filepath + '.tar.bz2', extract_dir=head)
        print('Done.')

def compress_model(dir_path):
    """Compresses a given directory using bz2 compression.

    Args:
        dir_path (str): path to directory to compress.

    Returns:
        True if compression completed without error.

    Raises:
        ValueError: if no directory at 'dir_path' exists or if 'dir_path'.tar.bz2 already exists.
    """
    # clean/normalize dir_path
    dir_path = os.path.abspath(os.path.normcase(os.path.normpath(dir_path)))

    output_filepath = '{}.tar.bz2'.format(dir_path)
    if os.path.exists(output_filepath):
        raise ValueError("{} already exists.".format(output_filepath))
    if not os.path.exists(dir_path):
        raise ValueError("File or directory at `dir_path` does not exist.")

    # create bz2 compressed directory, remove uncompressed directory
    root_dir = os.path.abspath(''.join(os.path.split(dir_path)[:-1]))
    base_dir = os.path.basename(dir_path)
    shutil.make_archive(base_name=dir_path, format='bztar', root_dir=root_dir, base_dir=base_dir)
    shutil.rmtree(dir_path)

    return True

def bin_to_txt(filepath, output_dir=os.getcwd()):
    """Converts word embeddings given in the binary C format (w2v) to a simple
    text format that can be used with Saber.

    Args:
        filepath (str): path to the word vectors file in binary C (w2v) format
        output_dir (str): path to save converted word vectors file (defaults to
            current working directory)
    """
    # bad practice, but requires gensim to be installed when using
    # SequenceProcessor if import statement is at the top of this file
    from gensim.models.keyedvectors import KeyedVectors

    # load word vectors provided in C binary format
    word_vectors = KeyedVectors.load_word2vec_format(filepath, binary=True)
    vocab = word_vectors.vocab

    # create a new filepath
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_filepath = os.path.join(output_dir, base_name + '.txt')

    # write contents of input_file to new file output_filepath in txt format
    with codecs.open(output_filepath, 'w+', encoding='utf-8') as out_file:
        for word in vocab:
            vector = word_vectors[word]
            out_file.write("%s %s\n" %(word, " ".join(str(v) for v in vector)))

    print('[INFO] Converted C binary file saved to {}'.format(output_filepath))

def get_pretrained_model_dir(config):
    """Returns a directory path to save a pretrained model.

    Returns a directory path to save a pretrained model based on `config.dataset_folder`. The
    folder which contains the saved model is named from each dataset name in `config.dataset_folder`
    joined by an underscore. The full path is
    `config.output_folder/constants.PRETRAINED_MODEL_DIR`/dataset_names

    Args:
        config (Config): Config object

    Returns:
        full path to save a pretrained model folder based on `config.dataset_folder`
    """
    ds_names = '_'.join([os.path.basename(ds) for ds in config.dataset_folder])
    return os.path.join(config.output_folder, constants.PRETRAINED_MODEL_DIR, ds_names)
