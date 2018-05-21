import os
import errno
import codecs

"""A collection of generic helper/utility functions."""

# https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist#273227
def make_dir(directory_filepath):
    """Creates a directory (directory_filepath) if it does not exist.

    Args:
        directory_filepath: filepath of directory to create
    """
    # create output directory if it does not exist
    try:
        os.makedirs(directory_filepath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

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
    word_vectors = KeyedVectors.load_word2vec_format(filepath, \
        binary=True)
    vocab = word_vectors.vocab

    # create a new filepath
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_filepath = os.path.join(output_dir, base_name + '.txt')

    # write contents of input_file to new file output_filepath in txt format
    with codecs.open(output_filepath, 'w+', encoding='utf-8') as f:
        for word in vocab:
            vector = word_vectors[word]
            f.write("%s %s\n" %(word, " ".join(str(v) for v in vector)))

    print('[INFO] Converted C binary file saved to {}'.format(output_filepath))
