import os
import errno

""" A collection of helper/utility functions. """

# https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist#273227
def make_dir(directory_filepath):
    """ Creates a directory (directory_filepath) if it does not exist.

    Args:
        directory_filepath: filepath of directory to create
    """
    # create output directory if it does not exist
    try:
        os.makedirs(directory_filepath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
