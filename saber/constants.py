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
PAD_VALUE = 0 # value of sequence pad

# train and test filename patterns
TRAIN_FILE_EXT = 'train.*'
# TEST_FILE_EXT = 'test.*'
