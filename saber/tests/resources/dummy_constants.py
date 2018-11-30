"""Constants used by the unit tests.
"""
import logging
import os

import numpy as np
from pkg_resources import resource_filename

from ... import constants

LOGGER = logging.getLogger(__name__)

# relative paths for test resources
PATH_TO_DUMMY_DATASET_1 = resource_filename(__name__, 'dummy_dataset_1')
PATH_TO_DUMMY_DATASET_2 = resource_filename(__name__, 'dummy_dataset_2')
PATH_TO_DUMMY_CONFIG = resource_filename(__name__, 'dummy_config.ini')
PATH_TO_DUMMY_EMBEDDINGS = resource_filename(__name__, 'dummy_word_embeddings/dummy_word_embeddings.txt')

######################################### DUMMY EMBEDDINGS #########################################

# for testing embeddings
DUMMY_TOKEN_MAP = {'<PAD>': 0, '<UNK>': 1, 'the': 2, 'quick': 3, 'brown': 4, 'fox': 5}
DUMMY_CHAR_MAP = {'<PAD>': 0, '<UNK>': 1, 'r': 2, 'u': 3, 'c': 4, 'f': 5, 'e': 6, 'o': 7, 'x': 8,
                  'h': 9, 'b': 10, 'n': 11, 'w': 12, 'i': 13, 't': 14, 'q': 15, 'k': 16}
DUMMY_EMBEDDINGS_INDEX = {
    'the': [0.15580128, -0.07108746, 0.055198, -0.14199848, 0.0005317868],
    'quick': [-0.011208724, 0.21213274, -0.17233513, -0.4401193, 0.13930725],
    'brown': [0.12754257, -0.07938199, 0.083904505, -0.24103324, 0.0084449835],
    'fox': [0.2947119, 0.14794342, 0.10318808, 0.09019197, -0.24244581]
}
DUMMY_EMBEDDINGS_MATRIX = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.15580128, -0.07108746, 0.055198, -0.14199848, 0.0005317868],
    [-0.011208724, 0.21213274, -0.17233513, -0.4401193, 0.13930725],
    [0.12754257, -0.07938199, 0.083904505, -0.24103324, 0.0084449835],
    [0.2947119, 0.14794342, 0.10318808, 0.09019197, -0.24244581]
])

########################################## DUMMY DATASET ##########################################

DUMMY_WORD_SEQ = np.array([
    ['Human', 'APC2', 'maps', 'to', 'chromosome', '19p13', '.'],
    ['The', 'absence', 'of', 'functional', 'C7', 'activity', 'could', 'not', 'be', 'accounted',
     'for', 'on', 'the', 'basis', 'of', 'an', 'inhibitor', '.'],
])
DUMMY_CHAR_SEQ = np.array([
    [['H', 'u', 'm', 'a', 'n'], ['A', 'P', 'C', '2'], ['m', 'a', 'p', 's'], ['t', 'o'],
     ['c', 'h', 'r', 'o', 'm', 'o', 's', 'o', 'm', 'e'], ['1', '9', 'p', '1', '3'], ['.']],
    [['T', 'h', 'e'], ['a', 'b', 's', 'e', 'n', 'c', 'e'], ['o', 'f'],
     ['f', 'u', 'n', 'c', 't', 'i', 'o', 'n', 'a', 'l'], ['C', '7'],
     ['a', 'c', 't', 'i', 'v', 'i', 't', 'y'], ['c', 'o', 'u', 'l', 'd'], ['n', 'o', 't'],
     ['b', 'e'], ['a', 'c', 'c', 'o', 'u', 'n', 't', 'e', 'd'], ['f', 'o', 'r'], ['o', 'n'],
     ['t', 'h', 'e'], ['b', 'a', 's', 'i', 's'], ['o', 'f'], ['a', 'n'],
     ['i', 'n', 'h', 'i', 'b', 'i', 't', 'o', 'r'], ['.']]])
DUMMY_TAG_SEQ = np.array([
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'B-DISO', 'I-DISO', 'I-DISO', 'E-DISO', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O'],
])
DUMMY_WORD_TYPES = ['Human', 'APC2', 'maps', 'to', 'chromosome', '19p13', '.', 'The', 'absence',
                    'of', 'functional', 'C7', 'activity', 'could', 'not', 'be', 'accounted',
                    'for', 'on', 'the', 'basis', 'an', 'inhibitor', constants.PAD, constants.UNK]
DUMMY_CHAR_TYPES = ['2', 's', 'c', 'T', 'd', 'e', 'H', 'h', 'a', 'b', 'v', 'C', 'm', 't', '9', 'p',
                    'r', '3', 'u', '.', 'o', '7', 'n', 'f', 'y', 'l', '1', 'i', 'A', 'P',
                    constants.PAD, constants.UNK]
DUMMY_TAG_TYPES = ['O', 'B-DISO', 'I-DISO', 'E-DISO', constants.PAD]

########################################### DUMMY CONFIG ###########################################

# Sections of the .ini file
CONFIG_SECTIONS = ['mode', 'data', 'model', 'training', 'advanced']

# Arg values before any processing
DUMMY_ARGS_NO_PROCESSING = {'model_name': 'MT-LSTM-CRF',
                            'save_model': 'False',
                            'dataset_folder': 'saber/tests/resources/dummy_dataset_1',
                            'output_folder': '../output',
                            'pretrained_model': '',
                            'pretrained_embeddings': ('saber/tests/resources/'
                                                      'dummy_word_embeddings/'
                                                      'dummy_word_embeddings.txt'),
                            'word_embed_dim': '200',
                            'char_embed_dim': '30',
                            'optimizer': 'nadam',
                            'activation': 'relu',
                            'learning_rate': '0.0',
                            'grad_norm': '1.0',
                            'decay': '0.0',
                            'dropout_rate': '0.3, 0.3, 0.1',
                            'batch_size': '32',
                            'k_folds': '2',
                            'epochs': '50',
                            'criteria': 'exact',
                            'verbose': 'False',
                            'debug': 'False',
                            'save_all_weights': 'False',
                            'tensorboard': 'False',
                            'replace_rare_tokens': 'False',
                            'load_all_embeddings': 'False',
                            'fine_tune_word_embeddings': 'False',
                            # TEMP
                            'variational_dropout': 'False',
                           }
# Final arg values when args provided in only config file
DUMMY_ARGS_NO_CLI_ARGS = {'model_name': 'mt-lstm-crf',
                          'save_model': False,
                          'dataset_folder': [PATH_TO_DUMMY_DATASET_1],
                          'output_folder': os.path.abspath('../output'),
                          'pretrained_model': '',
                          'pretrained_embeddings': PATH_TO_DUMMY_EMBEDDINGS,
                          'word_embed_dim': 200,
                          'char_embed_dim': 30,
                          'optimizer': 'nadam',
                          'activation': 'relu',
                          'learning_rate': 0.0,
                          'decay': 0.0,
                          'grad_norm': 1.0,
                          'dropout_rate': {'input': 0.3, 'output':0.3, 'recurrent': 0.1},
                          'batch_size': 32,
                          'k_folds': 2,
                          'epochs': 50,
                          'criteria': 'exact',
                          'verbose': False,
                          'debug': False,
                          'save_all_weights': False,
                          'tensorboard': False,
                          'replace_rare_tokens': False,
                          'load_all_embeddings': False,
                          'fine_tune_word_embeddings': False,
                          # TEMP
                          'variational_dropout': False,
                         }
# Final arg values when args provided in config file and from CLI
DUMMY_COMMAND_LINE_ARGS = {'optimizer': 'sgd',
                           'grad_norm': 1.0,
                           'learning_rate': 0.05,
                           'decay': 0.5,
                           'dropout_rate': [0.6, 0.6, 0.2],
                           # the dataset and embeddings are used for test purposes so they must
                           # point to the correct resources, this can be ensured by passing their
                           # paths here
                           'dataset_folder': [PATH_TO_DUMMY_DATASET_1],
                           'pretrained_embeddings': PATH_TO_DUMMY_EMBEDDINGS,
                          }
DUMMY_ARGS_WITH_CLI_ARGS = {'model_name': 'mt-lstm-crf',
                            'save_model': False,
                            'dataset_folder': [PATH_TO_DUMMY_DATASET_1],
                            'output_folder': os.path.abspath('../output'),
                            'pretrained_model': '',
                            'pretrained_embeddings': PATH_TO_DUMMY_EMBEDDINGS,
                            'word_embed_dim': 200,
                            'char_embed_dim': 30,
                            'optimizer': 'sgd',
                            'activation': 'relu',
                            'learning_rate': 0.05,
                            'decay': 0.5,
                            'grad_norm': 1.0,
                            'dropout_rate': {'input': 0.6, 'output': 0.6, 'recurrent': 0.2},
                            'batch_size': 32,
                            'k_folds': 2,
                            'epochs': 50,
                            'criteria': 'exact',
                            'verbose': False,
                            'debug': False,
                            'save_all_weights': False,
                            'tensorboard': False,
                            'replace_rare_tokens': False,
                            'load_all_embeddings': False,
                            'fine_tune_word_embeddings': False,
                            # TEMP
                            'variational_dropout': False,
                           }
########################################### WEB SERVICE ###########################################

DUMMY_ENTITIES = {'ANAT': False,
                  'CHED': True,
                  'DISO': False,
                  'LIVB': True,
                  'PRGE': True,
                  'TRIG': False,
                 }
