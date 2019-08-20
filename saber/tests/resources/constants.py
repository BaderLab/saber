"""Constants used by the unit tests.
"""
import os

import numpy as np
from pkg_resources import resource_filename

from ... import constants

# TODO (John): Eventually, get around to moving these to the top of their respective test files.

PATH_TO_CONLL2003_DATASET = resource_filename(__name__, 'conll2003_dataset')
PATH_TO_CONLL2004_DATASET = resource_filename(__name__, 'conll2004_dataset')
PATH_TO_DUMMY_DATASET_2 = resource_filename(__name__, 'dummy_dataset_2')
PATH_TO_DUMMY_CONFIG = resource_filename(__name__, 'dummy_config.ini')


####################################################################################################
# CoNLL2003 dataset
####################################################################################################


CoNLL2003_WORD_SEQ = np.array([
    ['Human', 'APC2', 'maps', 'to', 'chromosome', '19p13', '.'],
    ['The', 'absence', 'of', 'functional', 'C7', 'activity', 'could', 'not', 'be', 'accounted',
     'for', 'on', 'the', 'basis', 'of', 'an', 'inhibitor', '.'],
])
CoNLL2003_CHAR_SEQ = np.array([
    [['H', 'u', 'm', 'a', 'n'], ['A', 'P', 'C', '2'], ['m', 'a', 'p', 's'], ['t', 'o'],
     ['c', 'h', 'r', 'o', 'm', 'o', 's', 'o', 'm', 'e'], ['1', '9', 'p', '1', '3'], ['.']],
    [['T', 'h', 'e'], ['a', 'b', 's', 'e', 'n', 'c', 'e'], ['o', 'f'],
     ['f', 'u', 'n', 'c', 't', 'i', 'o', 'n', 'a', 'l'], ['C', '7'],
     ['a', 'c', 't', 'i', 'v', 'i', 't', 'y'], ['c', 'o', 'u', 'l', 'd'], ['n', 'o', 't'],
     ['b', 'e'], ['a', 'c', 'c', 'o', 'u', 'n', 't', 'e', 'd'], ['f', 'o', 'r'], ['o', 'n'],
     ['t', 'h', 'e'], ['b', 'a', 's', 'i', 's'], ['o', 'f'], ['a', 'n'],
     ['i', 'n', 'h', 'i', 'b', 'i', 't', 'o', 'r'], ['.']]])
CoNLL2003_ENT_SEQ = np.array([
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'B-DISO', 'I-DISO', 'I-DISO', 'E-DISO', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O'],
])
CoNLL2003_WORD_TYPES = ['Human', 'APC2', 'maps', 'to', 'chromosome', '19p13', '.', 'The', 'absence',
                        'of', 'functional', 'C7', 'activity', 'could', 'not', 'be', 'accounted',
                        'for', 'on', 'the', 'basis', 'an', 'inhibitor', constants.PAD,
                        constants.UNK]
CoNLL2003_CHAR_TYPES = ['2', 's', 'c', 'T', 'd', 'e', 'H', 'h', 'a', 'b', 'v', 'C', 'm', 't', '9',
                        'p', 'r', '3', 'u', '.', 'o', '7', 'n', 'f', 'y', 'l', '1', 'i', 'A', 'P',
                        constants.PAD, constants.UNK]
CoNLL2003_ENT_TYPES = ['O', 'B-DISO', 'I-DISO', 'E-DISO', constants.PAD]


####################################################################################################
# CoNLL2004 dataset
####################################################################################################


CoNLL2004_WORD_SEQ = np.array([
    ['Bette/Davis', 'was', 'born', 'Ruth/Elizabeth/Davis', 'on', 'April', '5', ',', '1908', ',',
     'in', 'Lowell/,/Mass', '.'],
    ['By', 'W./DALE/NELSON'],
    ['Associated/Press', 'Writer']
])

CoNLL2004_CHAR_SEQ = np.array([
    [['B', 'e', 't', 't', 'e', '/', 'D', 'a', 'v', 'i', 's'], ['w', 'a', 's'], ['b', 'o', 'r', 'n'],
     ['R', 'u', 't', 'h', '/', 'E', 'l', 'i', 'z', 'a', 'b', 'e', 't', 'h', '/', 'D', 'a', 'v', 'i',
      's'], ['o', 'n'], ['A', 'p', 'r', 'i', 'l'], ['5'], [','], ['1', '9', '0', '8'], [','],
     ['i', 'n'], ['L', 'o', 'w', 'e', 'l', 'l', '/', ',', '/', 'M', 'a', 's', 's'],
     ['.']],
    [['B', 'y'], ['W', '.', '/', 'D', 'A', 'L', 'E', '/', 'N', 'E', 'L', 'S', 'O', 'N']],
    [['A', 's', 's', 'o', 'c', 'i', 'a', 't', 'e', 'd', '/', 'P', 'r', 'e', 's', 's'], ['W', 'r',
     'i', 't', 'e', 'r']]
])

CoNLL2004_ENT_SEQ = np.array([
    ['S-Peop', 'O', 'O', 'S-Peop', 'O', 'S-Other', 'O', 'O', 'O', 'O', 'O', 'S-Loc', 'O'],
    ['O', 'S-Peop'],
    ['S-Org', 'O'],
])

CoNLL2004_REL_SEQ = [[[0, 11, 'Live_In'], [3, 11, 'Live_In']], [], []]

CoNLL2004_WORD_TYPES = ['Bette/Davis', 'was', 'born', 'Ruth/Elizabeth/Davis', 'on', 'April', '5',
                        ',', '1908', 'in', 'Lowell/,/Mass', '.', 'By', 'W./DALE/NELSON',
                        'Associated/Press', 'Writer', constants.PAD, constants.UNK]

CoNLL2004_CHAR_TYPES = ['o', 'S', 'b', 'u', 'D', 'w', 'v', 'O', 'n', ',', 't', '/', '0', 'W', 's',
                        '.', 'c', 'M', 'P', 'r', 'E', 'h', 'R', 'L', 'd', 'N', 'l', 'B', '8', 'e',
                        '1', 'z', 'A', '5', 'p', 'a', 'i', 'y', '9', constants.PAD, constants.UNK]

CoNLL2004_ENT_TYPES = ['O', 'S-Peop', 'S-Other', 'S-Loc', 'S-Org', constants.PAD]

CoNLL2004_REL_TYPES = ['Live_In', constants.NEG]


####################################################################################################
# Dummy config
####################################################################################################


# Sections of the .ini file
CONFIG_SECTIONS = ['mode', 'data', 'model', 'training', 'advanced']

# Arg values before any processing
DUMMY_ARGS_NO_PROCESSING = {'model_name': 'bert-ner',
                            'save_model': 'False',
                            'dataset_folder': 'saber/tests/resources/conll2003_dataset',
                            'dataset_reader': 'CoNLL2003DatasetReader',
                            'output_folder': '../output',
                            'pretrained_model': '',
                            'word_embed_dim': '200',
                            'char_embed_dim': '30',
                            'optimizer': 'nadam',
                            'activation': 'relu',
                            'learning_rate': '0.0',
                            'grad_norm': '1.0',
                            'decay': '0.0',
                            'dropout_rate': '0.1',
                            'batch_size': '32',
                            'validation_split': '0.0',
                            'k_folds': '2',
                            'epochs': '50',
                            'criteria': 'exact',
                            'verbose': 'False',
                            'debug': 'False',
                            'save_all_weights': 'False',
                            'tensorboard': 'False',
                            'replace_rare_tokens': 'False',
                            # TEMP
                            'variational_dropout': 'False',
                            }
# Final arg values when args provided in only config file
DUMMY_ARGS_NO_CLI_ARGS = {'model_name': 'bert-ner',
                          'save_model': False,
                          'dataset_folder': [PATH_TO_CONLL2003_DATASET],
                          'dataset_reader': 'conll2003datasetreader',
                          'output_folder': os.path.abspath('../output'),
                          'pretrained_model': '',
                          'word_embed_dim': 200,
                          'char_embed_dim': 30,
                          'optimizer': 'nadam',
                          'activation': 'relu',
                          'learning_rate': 0.0,
                          'decay': 0.0,
                          'grad_norm': 1.0,
                          'dropout_rate': 0.1,
                          'batch_size': 32,
                          'validation_split': 0.0,
                          'k_folds': 2,
                          'epochs': 50,
                          'criteria': 'exact',
                          'verbose': False,
                          'debug': False,
                          'save_all_weights': False,
                          'tensorboard': False,
                          'replace_rare_tokens': False,
                          # TEMP
                          'variational_dropout': False,
                          }
# Final arg values when args provided in config file and from CLI
DUMMY_COMMAND_LINE_ARGS = {'optimizer': 'sgd',
                           'grad_norm': 1.0,
                           'learning_rate': 0.05,
                           'decay': 0.5,
                           'dropout_rate': 0.6,
                           # the datasets are used for test purposes so they must
                           # point to the correct resources, this can be ensured by passing their
                           # paths here
                           'dataset_folder': [PATH_TO_CONLL2003_DATASET],
                           }
DUMMY_ARGS_WITH_CLI_ARGS = {'model_name': 'bert-ner',
                            'save_model': False,
                            'dataset_folder': [PATH_TO_CONLL2003_DATASET],
                            'dataset_reader': 'conll2003datasetreader',
                            'output_folder': os.path.abspath('../output'),
                            'pretrained_model': '',
                            'word_embed_dim': 200,
                            'char_embed_dim': 30,
                            'optimizer': 'sgd',
                            'activation': 'relu',
                            'learning_rate': 0.05,
                            'decay': 0.5,
                            'grad_norm': 1.0,
                            'dropout_rate': 0.6,
                            'batch_size': 32,
                            'validation_split': 0.0,
                            'k_folds': 2,
                            'epochs': 50,
                            'criteria': 'exact',
                            'verbose': False,
                            'debug': False,
                            'save_all_weights': False,
                            'tensorboard': False,
                            'replace_rare_tokens': False,
                            # TEMP
                            'variational_dropout': False,
                            }


####################################################################################################
# Web-service
####################################################################################################


DUMMY_ENTITIES = {'ANAT': False,
                  'CHED': True,
                  'DISO': False,
                  'LIVB': True,
                  'PRGE': True,
                  'TRIG': False,
                  }
