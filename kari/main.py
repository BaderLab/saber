import os
import argparse
import pandas as pd
import numpy as np

from dataset import Dataset
from utils import SentenceGetter
from specify import *

from keras_contrib.layers import CRF
from keras.models import Model, Input
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

'''
# Notes
- RNNs are tricky. Choice of batch size is important, choice of loss and
 optimizer is critical, etc. Some configurations won't converge.
-  LSTM loss decrease patterns during training can be quite different from what
you see with CNNs/MLPs/etc.
'''

def load_parameters(parameters_filepath):
    """ Load parameters from ini file is specificed.

    Loads parameters from the ini file if specified, take into account any
    command line argument, and ensure that each parameter is cast to the correct
    type. Command line arguments take precedence over parameters specified in
    the parameter file.

    Args:
        parameters_filepath: path to ini file containing the parameters

    """
    pass

def parse_arguments():
    """ Parse command line arguments passed with call to <>.

    Returns:
        <>

    """

    parser = argparse.ArgumentParser(description='<> CLI')


    parser.add_argument('--parameters_filepath', required=False,
                        default=os.path.join('.', 'parameters.ini'),
                        help = '''path to the .ini file containing the parameters.
                        Defaults to 'parameters.ini''')

    return parser.parse_args()


def main():
    """ <> main method.

    Args:
        parameters_filepath
        output_folder
    """

    parse_arguments()

dataset = pd.read_csv('/Users/johngiorgi/Documents/Masters/Class/natural_language_computing/project/LSTM_CRF/datasets/NCBI_disease_train.tsv', sep='\t', header=None, names=['Word', 'Tag'], encoding="latin1")
dataset


if __name__ == '__main__':

    dataset = Dataset('../../datasets/NCBI_disease_train.tsv')
    dataset.load_dataset()

    # print(dataset.word_idx_sequence)

    # split the data set into train/test
    X_tr, X_te, y_tr, y_te = train_test_split(dataset.word_idx_sequence,
        dataset.tag_idx_sequence, test_size=0.1)

    model, crf = specify(dataset.word_type_count, dataset.tag_type_count,
        max_len=dataset.max_seq_len)

    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    history = model.fit(X_tr, np.array(y_tr), batch_size=10, epochs=10,
                        validation_split=0.1, verbose=1)

    hist = pd.DataFrame(history.history)
