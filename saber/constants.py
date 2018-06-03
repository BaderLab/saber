"""Collection of constants used by Saber."""

# colours for use with displacy
COLOURS = {'PRGE': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)',
           'CHED': 'linear-gradient(90deg, #1DE9B6, #A7FFEB)',
           'LIVB': 'linear-gradient(90deg, #FF4081, #F8BBD0)',
           'CL': 'linear-gradient(90deg, #00E5FF, #84FFFF)',
          }
# entity options for use with displacy
OPTIONS = {# 'ents': ['PRGE', 'CHED', 'LIVB', 'CL', 'SO', 'GO'],
           'colors': COLOURS
          }

UNK = '<UNK>' # out-of-vocabulary token
PAD = '<PAD>' # sequence pad token
START = '<START>' # start-of-sentence token
END = '<END>' # end-of-sentence token

OUTSIDE_TAG = 'O' # 'outside' tag of the IOB, BIO, and IOBES tag formats

PAD_VALUE = 0 # value of sequence pad
NUM_RARE = 1 # tokens that occur less than NUM_RARE times are replaced UNK

# mapping of special tokens to contants
initial_mapping_words = {PAD: 0, UNK: 1}
initial_mapping_tags = {PAD: 0}

# train and test filename patterns
TRAIN_FILE_EXT = 'train.*'
# TEST_FILE_EXT = 'test.*'

# relative path to pretrained model directory
PRETRAINED_MODEL_BASE_DIR = '../pretrained_models'
