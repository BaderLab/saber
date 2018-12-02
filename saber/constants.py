"""Collection of constants used by Saber.
"""
from pkg_resources import resource_filename

__version__ = '0.0.1'

# DISPLACY OPTIONS
# entity colours
COLOURS = {'PRGE': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)',
           'DISO': 'linear-gradient(90deg, #ef9a9a, #f44336)',
           'CHED': 'linear-gradient(90deg, #1DE9B6, #A7FFEB)',
           'LIVB': 'linear-gradient(90deg, #FF4081, #F8BBD0)',
           'CL': 'linear-gradient(90deg, #00E5FF, #84FFFF)',
          }
# entity options
OPTIONS = {'colors': COLOURS}

# SPECIAL TOKENS
UNK = '<UNK>' # out-of-vocabulary token
PAD = '<PAD>' # sequence pad token
START = '<START>' # start-of-sentence token
END = '<END>' # end-of-sentence token
OUTSIDE_TAG = 'O' # 'outside' tag of the IOB, BIO, and IOBES tag formats

# MISC.
PAD_VALUE = 0 # value of sequence pad
NUM_RARE = 1 # tokens that occur less than NUM_RARE times are replaced UNK
# mapping of special tokens to contants
INITIAL_MAPPING = {'word': {PAD: 0, UNK: 1}, 'tag':  {PAD: 0}}
# keys into dictionaries containing information for different partitions of a dataset
PARTITIONS = ['train', 'valid', 'test']

# FILEPATHS / FILENAMES
# train, valid and test filename patterns
TRAIN_FILE = 'train.*'
VALID_FILE = 'valid.*'
TEST_FILE = 'test.*'
# pre-trained models
ENTITIES = {'ANAT': False,
            'CHED': False,
            'DISO': True,
            'LIVB': False,
            'PRGE': True,
            'TRIG': False}
PRETRAINED_MODELS = [ent for ent, value in ENTITIES.items() if value]
# relative path to pre-trained model directory
PRETRAINED_MODEL_DIR = resource_filename(__name__, 'pretrained_models')
MODEL_FILENAME = 'model_params.json'
WEIGHTS_FILENAME = 'model_weights.hdf5'
ATTRIBUTES_FILENAME = 'attributes.pickle'
CONFIG_FILENAME = 'config.ini'

# MODEL SETTINGS
# batch size to use when performing model prediction
PRED_BATCH_SIZE = 256
# max length of a sentence
MAX_SENT_LEN = 100
# max length of a character sequence (word)
MAX_CHAR_LEN = 25
# number of units in the LSTM layers
UNITS_WORD_LSTM = 200
UNITS_CHAR_LSTM = 200
UNITS_DENSE = UNITS_WORD_LSTM // 2
# possible models
MODEL_NAMES = ['mt-lstm-crf',]

# RESTful API
# endpoint for Entrez Utilities Web Service API
EUTILS_API_ENDPOINT = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?retmode=xml&db='
                       'pubmed&id=')
# CONFIG
CONFIG_ARGS = ['model_name', 'save_model', 'dataset_folder', 'output_folder',
               'pretrained_model', 'pretrained_embeddings', 'word_embed_dim',
               'char_embed_dim', 'optimizer', 'activation', 'learning_rate', 'decay', 'grad_norm',
               'dropout_rate', 'batch_size', 'k_folds', 'epochs', 'criteria', 'verbose',
               'debug', 'save_all_weights', 'tensorboard', 'replace_rare_tokens',
               'load_all_embeddings', 'fine_tune_word_embeddings', 'variational_dropout']
