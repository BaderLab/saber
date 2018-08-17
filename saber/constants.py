"""Collection of constants used by Saber.
"""

__version__ = '0.0.1'

# DISPLACY OPTIONS
# entity colours
COLOURS = {'PRGE': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)',
           'CHED': 'linear-gradient(90deg, #1DE9B6, #A7FFEB)',
           'LIVB': 'linear-gradient(90deg, #FF4081, #F8BBD0)',
           'CL': 'linear-gradient(90deg, #00E5FF, #84FFFF)',
          }
# entity options
# OPTIONS = {'ents': ['PRGE', 'CHED', 'LIVB', 'CL', 'SO', 'GO'], 'colors': COLOURS}
OPTIONS = {'colors': COLOURS}

# SPECIAL TOKENS
UNK = '<UNK>' # out-of-vocabulary token
PAD = '<PAD>' # sequence pad token
START = '<START>' # start-of-sentence token
END = '<END>' # end-of-sentence token
OUTSIDE_TAG = 'O' # 'outside' tag of the IOB, BIO, and IOBES tag formats

PAD_VALUE = 0 # value of sequence pad
NUM_RARE = 1 # tokens that occur less than NUM_RARE times are replaced UNK

# mapping of special tokens to contants
INITIAL_MAPPING = {'word': {PAD: 0, UNK: 1}, 'tag':  {PAD: 0}}

# keys into dictionaries containing information for different partitions of a
# dataset
PARTITIONS = ['train', 'valid', 'test']

# FILEPATHS
# train, valid and test filename patterns
TRAIN_FILE = 'train.*'
VALID_FILE = 'valid.*'
TEST_FILE = 'test.*'
# relative path to pretrained model directory
PRETRAINED_MODEL_DIR = 'pretrained_models'
MODEL_FILEPATH = 'model_params.json'
WEIGHTS_FILEPATH = 'model_weights.hdf5'
ATTRIBUTES_FILEPATH = 'attributes.pickle'

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
MODELS = ['mt-lstm-crf',]

# RESTful API
# endpoint for Entrez Utilities Web Service API
EUTILS_API_ENDPOINT = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?retmode=xml&db='
                       'pubmed&id=')

# CONFIG

CONFIG_ARGS = ['model_name', 'train_model', 'dataset_folder', 'output_folder',
               'pretrained_model_weights', 'pretrained_embeddings', 'word_embed_dim',
               'char_embed_dim', 'optimizer', 'activation', 'learning_rate', 'decay', 'grad_norm',
               'dropout_rate', 'batch_size', 'k_folds', 'epochs', 'criteria', 'verbose',
               'debug', 'tensorboard', 'replace_rare_tokens', 'fine_tune_word_embeddings',
               'variational_dropout']
