"""Collection of constants used by Saber.
"""
from pkg_resources import resource_filename

__version__ = '0.1.0a0'

####################################################################################################
# SpaCy options
####################################################################################################

# The spaCy model to load. Defaults to the large sized English model.
SPACY_MODEL = 'en_core_web_md'

# Entity colours
COLOURS = {'PRGE': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)',
           'DISO': 'linear-gradient(90deg, #FF5252, #FFCDD2)',
           'CHED': 'linear-gradient(90deg, #1DE9B6, #A7FFEB)',
           'LIVB': 'linear-gradient(90deg, #FF4081, #F8BBD0)',
           'CL': 'linear-gradient(90deg, #00E5FF, #84FFFF)',
           }
# Entity options
OPTIONS = {'colors': COLOURS}

####################################################################################################
# NeuralCoref options
####################################################################################################

# Greedyness of NeuralCoref. See here: https://github.com/huggingface/neuralcoref#parameters
NEURALCOREF_GREEDYNESS = 0.40

####################################################################################################
# Special values
####################################################################################################

# Special tokens
UNK = '[UNK]'  # out-of-vocabulary token
PAD = '[PAD]'  # sequence pad token
START = '[START]'  # start-of-sentence token
END = '[END]'  # end-of-sentence token
OUTSIDE = 'O'  # 'outside' tag of the IOB, BIO, and IOBES tag formats
NEG = 'NEG'  # "no relation" class for relation classification

# BERT-related tokens
WORDPIECE = 'X'  # special tag used by BERTs wordpiece tokenizer
CLS = '[CLS]'  # special BERT classification token
SEP = '[SEP]'  # special BERT sequence seperator token

# tags representing the end of a chunk for BIO, BIOES and BILOU tag sets
CHUNK_END_TAGS = ['L-', 'U-', 'E-', 'S-']

# Special integer constants
RANDOM_STATE = 42  # random seed
PAD_VALUE = 0  # value of sequence pad
UNK_VALUE = 1  # value of unknown pad
NEG_VALUE = 0  # value of NEG relation class
TOK_MAP_PAD = -100  # value of original token map (used for WordPiece tokenized text) pad
NUM_RARE = 1  # tokens that occur less than NUM_RARE times are replaced UNK

# Mapping of special tokens to contants
INITIAL_MAPPING = {'word': {PAD: PAD_VALUE, UNK: UNK_VALUE},
                   'ent':  {PAD: PAD_VALUE},
                   'rel': {NEG: NEG_VALUE}}

####################################################################################################
# Filepaths / filenames
####################################################################################################

# Keys into dictionaries containing information for different partitions of a dataset
PARTITIONS = ['train', 'valid', 'test']

# Train, valid and test filename patterns
TRAIN_FILE = 'train.*'
VALID_FILE = 'valid.*'
TEST_FILE = 'test.*'

PRETRAINED_MODEL_DIR = resource_filename(__name__, 'pretrained_models')
PRETRAINED_MODEL_FILENAME = 'model.bin'
ATTRIBUTES_FILENAME = 'attributes.pickle'
CONFIG_FILENAME = 'config.ini'

# Pre-trained models
ENTITIES = {'ANAT': False,
            'CHED': True,
            'DISO': True,
            'LIVB': False,
            'PRGE': True,
            'TRIG': False}

# Google Drive File IDs for the pre-trained models
PRETRAINED_MODELS = {
    'PRGE': '1xOmxpgNjQJK8OJSvih9wW5AITGQX6ODT',
    'DISO': '1qmrBuqz75KM57Ug5MiDBfp0d5H3S_5ih',
    'CHED': '13s9wvu3Mc8fG73w51KD8RArA31vsuL1c',
    'biobert_v1.1_pubmed': '1jI1HyzMzSShjHfeO1pSmw5su8R6p5Vsv'
}


# TODO (John): Most of these values should be added to the config once that API is settled
####################################################################################################
# Model Settings
####################################################################################################

# Batch size to use when performing model prediction
PRED_BATCH_SIZE = 256
# Max length of a sentence, set to None to use the length of the longest sentence
MAX_SENT_LEN = 256
# Max length of a character sequence (word), set to None to use the length of the longest word
MAX_CHAR_LEN = 25

# Possible models
MODEL_NAMES = ['bert-ner', 'bert-ner-re']

# Which pre-trained BERT model to use
PYTORCH_BERT_MODEL = 'biobert_v1.1_pubmed'

####################################################################################################
# EXTRACT 2.0 API
####################################################################################################
# arguments passed in a get request to the EXTRACT 2.0 API to specify entity type
ENTITY_TYPES = {'CHED': -1, 'DISO': -26, 'LIVB': -2}
# the namespaces of the external resources that EXTRACT 2.0 grounds too
NAMESPACES = {'CHED': 'PubChem Compound',
              'DISO': 'Disease Ontology',
              'LIVB': 'NCBI Taxonomy',
              'PRGE': 'STRING',
              }

####################################################################################################
# RESTful API
####################################################################################################
# endpoint for Entrez Utilities Web Service API
EUTILS_API_ENDPOINT = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?retmode=xml&db='
                       'pubmed&id=')
# CONFIG
CONFIG_ARGS = ['model_name', 'save_model', 'dataset_folder', 'output_folder',
               'pretrained_model', 'optimizer', 'activation', 'learning_rate', 'decay', 'grad_norm',
               'dropout_rate', 'batch_size', 'k_folds', 'epochs', 'criteria', 'verbose', 'debug',
               'save_all_weights', 'tensorboard', 'replace_rare_tokens', 'validation_split',
               'dataset_reader']
