"""A collection of PyTest fixtures used for Sabers unit test (saber/tests/test_*.py)
"""
import os

import pytest
import spacy
from keras.utils import to_categorical
from pytorch_pretrained_bert import BertTokenizer

from .. import constants
from ..config import Config
from ..dataset import CoNLL2003DatasetReader
from ..dataset import CoNLL2004DatasetReader
from ..dataset import Dataset
from ..embeddings import Embeddings
from ..metrics import Metrics
from ..models.base_model import BaseKerasModel
from ..models.base_model import BaseModel
from ..models.base_model import BasePyTorchModel
from ..models.bert_for_joint_ner_and_rc import BertForJointNERAndRE
from ..models.bert_for_ner import BertForNER
from ..models.bilstm_crf import BiLSTMCRF
from ..preprocessor import Preprocessor
from ..saber import Saber
from ..utils import data_utils
from ..utils import model_utils
from ..utils import text_utils
from .resources.constants import DUMMY_COMMAND_LINE_ARGS
from .resources.constants import DUMMY_TOKEN_MAP
from .resources.constants import PATH_TO_CONLL2003_DATASET
from .resources.constants import PATH_TO_CONLL2004_DATASET
from .resources.constants import PATH_TO_DUMMY_CONFIG
from .resources.constants import PATH_TO_DUMMY_DATASET_2
from .resources.constants import PATH_TO_DUMMY_EMBEDDINGS


####################################################################################################
# Generic
####################################################################################################


@pytest.fixture(scope='session')
def dummy_dir(tmpdir_factory):
    """Returns the path to a temporary directory.
    """
    dummy_dir = tmpdir_factory.mktemp('dummy_dir')
    return dummy_dir.strpath


####################################################################################################
# Config
####################################################################################################


@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object."""
    return Config(PATH_TO_DUMMY_CONFIG)


@pytest.fixture
def dummy_config_cli_args():
    """Returns an instance of a config.config.Config object after parsing the dummy config file with
    command line interface (CLI) args."""
    # parse the dummy config, leave cli false and instead pass command line args manually
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    # this is a bit of a hack, but need to simulate providing commands at the command line
    dummy_config.cli_args = DUMMY_COMMAND_LINE_ARGS
    dummy_config.harmonize_args(DUMMY_COMMAND_LINE_ARGS)

    return dummy_config


@pytest.fixture
def dummy_config_compound_dataset():
    """Returns an instance of a `Config` after parsing the dummy config file. Ensures that
    `replace_rare_tokens` argument is False.
    """
    compound_dataset = [PATH_TO_CONLL2003_DATASET, PATH_TO_DUMMY_DATASET_2]
    cli_arguments = {'dataset_folder': compound_dataset}
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config.harmonize_args(cli_arguments)

    return dummy_config


####################################################################################################
# Dataset
####################################################################################################


@pytest.fixture
def dataset_no_dataset_folder():
    return Dataset(totally_arbitrary='arbitrary')


@pytest.fixture
def dataset():
    return Dataset(dataset_folder=PATH_TO_CONLL2003_DATASET,
                   totally_arbitrary='arbitrary')


@pytest.fixture
def dummy_dataset_2():
    """Returns a single dummy CoNLL2003DatasetReader instance after calling
    `CoNLL2003DatasetReader.load()`.
    """
    # Don't replace rare tokens for the sake of testing
    dataset = CoNLL2003DatasetReader(dataset_folder=PATH_TO_DUMMY_DATASET_2,
                                     replace_rare_tokens=False)
    dataset.load()

    return dataset


@pytest.fixture
def dummy_compound_dataset(dummy_config):
    """
    """
    dummy_config.dataset_folder = [PATH_TO_CONLL2003_DATASET, PATH_TO_DUMMY_DATASET_2]
    dummy_config.replace_rare_tokens = False
    dataset = data_utils.load_compound_dataset(dummy_config)

    return dataset


@pytest.fixture(scope='session')
def dummy_dataset_paths(tmpdir_factory):
    """Creates and returns the path to a temporary dataset folder, and train, valid, test files.
    """
    # create a dummy dataset folder
    dummy_dir = tmpdir_factory.mktemp('dummy_dataset')
    # create train, valid and train partitions in this folder
    train_file = dummy_dir.join('train.tsv')
    train_file.write('arbitrary')  # need to write content or else the file wont exist
    valid_file = dummy_dir.join('valid.tsv')
    valid_file.write('arbitrary')
    test_file = dummy_dir.join('test.tsv')
    test_file.write('arbitrary')

    return dummy_dir.strpath, train_file.strpath, valid_file.strpath, test_file.strpath


@pytest.fixture(scope='session')
def dummy_dataset_paths_no_valid(tmpdir_factory):
    """Creates and returns the path to a temporary dataset folder, and train, and test files.
    """
    # create a dummy dataset folder
    dummy_dir = tmpdir_factory.mktemp('dummy_dataset')
    # create train, valid and train partitions in this folder
    train_file = dummy_dir.join('train.tsv')
    train_file.write('arbitrary')  # need to write content or else the file wont exist
    test_file = dummy_dir.join('test.tsv')
    test_file.write('arbitrary')

    return dummy_dir.strpath, train_file.strpath, test_file.strpath


####################################################################################################
# CoNLL2003DatasetReader
####################################################################################################


@pytest.fixture
def conll2003datasetreader_no_dataset_folder():
    """Returns an empty single dummy CoNLL2003DatasetReader instance.
    """
    return CoNLL2003DatasetReader(totally_arbitrary='arbitrary')


@pytest.fixture
def conll2003datasetreader():
    """Returns an empty single dummy CoNLL2003DatasetReader instance.
    """
    return CoNLL2003DatasetReader(dataset_folder=PATH_TO_CONLL2003_DATASET,
                                  totally_arbitrary='arbitrary')


@pytest.fixture
def conll2003datasetreader_load():
    """Returns a single CoNLL2003DatasetReader instance after calling `load()`.
    """
    # Don't replace rare tokens for the sake of testing
    dataset = CoNLL2003DatasetReader(dataset_folder=PATH_TO_CONLL2003_DATASET)
    dataset.load()

    return dataset


####################################################################################################
# CoNLL2004DatasetReader
####################################################################################################


@pytest.fixture
def conll2004datasetreader_no_dataset_folder():
    """Returns an empty single dummy CoNLL2003DatasetReader instance.
    """
    return CoNLL2004DatasetReader(totally_arbitrary='arbitrary')


@pytest.fixture
def conll2004datasetreader():
    """Returns an empty single dummy CoNLL2003DatasetReader instance.
    """
    return CoNLL2004DatasetReader(dataset_folder=PATH_TO_CONLL2004_DATASET,
                                  totally_arbitrary='arbitrary')


@pytest.fixture
def conll2004datasetreader_load():
    """Returns a single CoNLL2003DatasetReader instance after calling `load()`.
    """
    # Don't replace rare tokens for the sake of testing
    dataset = CoNLL2004DatasetReader(dataset_folder=PATH_TO_CONLL2004_DATASET)
    dataset.load()

    return dataset


####################################################################################################
# Embeddings
####################################################################################################


@pytest.fixture
def dummy_embeddings(conll2003datasetreader_load):
    """Returns an instance of an `Embeddings()` object AFTER the `.load()` method is called.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS,
                            token_map=conll2003datasetreader_load.idx_to_tag)
    embeddings.load(binary=False)  # txt file format is easier to test
    return embeddings


@pytest.fixture
def dummy_embedding_idx():
    """Returns embedding index from call to `Embeddings._prepare_embedding_index()`.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=DUMMY_TOKEN_MAP)
    embedding_idx = embeddings._prepare_embedding_index(binary=False)
    return embedding_idx


@pytest.fixture
def dummy_embedding_matrix_and_type_to_idx():
    """Returns the `embedding_matrix` and `type_to_index` objects from call to
    `Embeddings._prepare_embedding_matrix(load_all=False)`.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=DUMMY_TOKEN_MAP)
    embedding_idx = embeddings._prepare_embedding_index(binary=False)
    embeddings.num_found = len(embedding_idx)
    embeddings.dimension = len(list(embedding_idx.values())[0])
    embedding_matrix, type_to_idx = embeddings._prepare_embedding_matrix(embedding_idx,
                                                                         load_all=False)
    embeddings.num_embed = embedding_matrix.shape[0]  # num of embedded words

    return embedding_matrix, type_to_idx


@pytest.fixture
def dummy_embedding_matrix_and_type_to_idx_load_all():
    """Returns the embedding matrix and type to index objects from call to
    `Embeddings._prepare_embedding_matrix(load_all=True)`.
    """
    # this should be different than DUMMY_TOKEN_MAP for a reliable test
    test = {"This": 0, "is": 1, "a": 2, "test": 3}

    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=test)
    embedding_idx = embeddings._prepare_embedding_index(binary=False)
    embeddings.num_found = len(embedding_idx)
    embeddings.dimension = len(list(embedding_idx.values())[0])
    embedding_matrix, type_to_idx = embeddings._prepare_embedding_matrix(embedding_idx,
                                                                         load_all=True)
    embeddings.num_embed = embedding_matrix.shape[0]  # num of embedded words

    return embedding_matrix, type_to_idx


@pytest.fixture
def dummy_embeddings_before_load():
    """Returns an instance of an Embeddings() object BEFORE the `Embeddings.load()` method is
    called.
    """
    return Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS,
                      token_map=DUMMY_TOKEN_MAP,
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')


@pytest.fixture
def dummy_embeddings_after_load():
    """Returns an instance of an Embeddings() object AFTER `Embeddings.load(load_all=False)` is
    called.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=DUMMY_TOKEN_MAP)
    embeddings.load(binary=False, load_all=False)  # txt file format is easier to test
    return embeddings


@pytest.fixture
def dummy_embeddings_after_load_with_load_all():
    """Returns an instance of an Embeddings() object AFTER `Embeddings.load(load_all=True)` is
    called.
    """
    # this should be different than DUMMY_TOKEN_MAP for a reliable test
    test = {"This": 0, "is": 1, "a": 2, "test": 3}

    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=test)
    embeddings.load(binary=False, load_all=True)  # txt file format is easier to test
    return embeddings


####################################################################################################
# Saber
####################################################################################################


@pytest.fixture
def saber_blank(dummy_config):
    """Returns instance of `Saber` initialized with the dummy config file and no dataset.
    """
    return Saber(config=dummy_config,
                 # to test passing of arbitrary keyword args to constructor
                 totally_arbitrary='arbitrary')


@pytest.fixture
def saber_single_dataset(dummy_config):
    """Returns instance of `Saber` initialized with the dummy config file and a single dataset.
    """
    saber = Saber(config=dummy_config)
    saber.load_dataset(directory=PATH_TO_CONLL2003_DATASET)

    return saber


@pytest.fixture
def saber_compound_dataset(dummy_config_compound_dataset):
    """Returns an instance of `Saber` initialized with the dummy config file and a compound dataset.
    The compound dataset is just two copies of the dataset, this makes writing tests much
    simpler.
    """
    compound_dataset = [PATH_TO_CONLL2003_DATASET, PATH_TO_CONLL2003_DATASET]
    saber = Saber(config=dummy_config_compound_dataset)
    saber.load_dataset(directory=compound_dataset)

    return saber


@pytest.fixture
def saber_bilstm_crf_model(dummy_config):
    """Returns an instance of `Saber` initialized with the dummy config file, a single dataset
    a Keras model."""
    saber = Saber(config=dummy_config)
    saber.load_dataset(directory=PATH_TO_CONLL2003_DATASET)
    saber.build(model_name='bilstm-crf-ner')

    return saber


@pytest.fixture
def saber_bert_for_ner_model(dummy_config):
    """Returns an instance of `Saber` initialized with the dummy config file, a single dataset
    a Keras model."""
    saber = Saber(config=dummy_config)
    saber.load_dataset(directory=PATH_TO_CONLL2003_DATASET)
    saber.build(model_name='bert-ner')

    return saber


@pytest.fixture
def saber_mt_bilstm_crf_model(dummy_config_compound_dataset):
    """Returns an instance of `Saber` initialized with the dummy config file, a single dataset
    a Keras model."""
    saber = Saber(config=dummy_config_compound_dataset)
    saber.load_dataset(directory=[PATH_TO_CONLL2003_DATASET, PATH_TO_DUMMY_DATASET_2])
    saber.build(model_name='bilstm-crf-ner')

    return saber


@pytest.fixture
def saber_mt_bert_for_ner_model(dummy_config_compound_dataset):
    """Returns an instance of `Saber` initialized with the dummy config file, a single dataset
    a Keras model."""
    saber = Saber(config=dummy_config_compound_dataset)
    saber.load_dataset(directory=[PATH_TO_CONLL2003_DATASET, PATH_TO_DUMMY_DATASET_2])
    saber.build(model_name='bert-ner')

    return saber


@pytest.fixture
def saber_single_dataset_embeddings(dummy_config):
    """Returns instance of `Saber` initialized with the dummy config file, a single dataset and
    embeddings.
    """
    saber = Saber(config=dummy_config)
    saber.load_dataset(directory=PATH_TO_CONLL2003_DATASET)
    saber.load_embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, binary=False)

    return saber


@pytest.fixture
def saber_saved_model(dummy_dir, dummy_config):
    """Returns a tuple containing an instance of a `Saber` object after `save()` was called and
    its models wiped (`saber.models = []`), the models and datasets it was saved with, and the
    directory where the model was saved.
    """
    saber = Saber(config=dummy_config)
    saber.load_dataset(directory=PATH_TO_CONLL2003_DATASET)
    saber.build()

    model, dataset = saber.models[-1], saber.datasets[-1]

    directory = saber.save(directory=dummy_dir)

    saber.models = []

    return saber, model, dataset, directory


####################################################################################################
# Model training
####################################################################################################


@pytest.fixture
def dummy_output_dir(tmpdir, dummy_config):
    """Returns list of output directories."""
    # make sure top-level directory is the pytest tmpdir
    dummy_config.output_folder = tmpdir.strpath
    output_dirs = model_utils.prepare_output_directory(dummy_config)

    return output_dirs


@pytest.fixture
def dummy_training_data(conll2003datasetreader_load):
    """Returns training data from `conll2003datasetreader_load`.
    """
    training_data = {
        'train': {
            'x': [conll2003datasetreader_load.idx_seq['train']['word'],
                  conll2003datasetreader_load.idx_seq['train']['char']],
            'y': to_categorical(conll2003datasetreader_load.idx_seq['train']['ent'])
        },
        'valid': None,
        'test': None,
    }

    # A list of lists which represents data for each fold (inner list) of each dataset (outer list)
    return [[training_data]]


####################################################################################################
# BaseModel
####################################################################################################


@pytest.fixture
def base_model(dummy_config, conll2003datasetreader_load):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BaseModel(config=dummy_config,
                      datasets=[conll2003datasetreader_load],
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_base_model(dummy_config_compound_dataset, conll2003datasetreader_load, dummy_dataset_2):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BaseModel(config=dummy_config_compound_dataset,
                      datasets=[conll2003datasetreader_load, dummy_dataset_2],
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def base_keras_model(dummy_config, conll2003datasetreader_load):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BaseKerasModel(config=dummy_config,
                           datasets=[conll2003datasetreader_load],
                           # to test passing of arbitrary keyword args to constructor
                           totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_base_keras_model(dummy_config_compound_dataset, conll2003datasetreader_load, dummy_dataset_2):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BaseKerasModel(config=dummy_config_compound_dataset,
                           datasets=[conll2003datasetreader_load, dummy_dataset_2],
                           # to test passing of arbitrary keyword args to constructor
                           totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def base_keras_model_embeddings(dummy_config, conll2003datasetreader_load, dummy_embeddings):
    """Returns an instance of BiLSTMCRF initialized with the default configuration file and
    loaded embeddings"""
    model = BaseKerasModel(config=dummy_config,
                           datasets=[conll2003datasetreader_load],
                           embeddings=dummy_embeddings,
                           # to test passing of arbitrary keyword args to constructor
                           totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def base_pytorch_model(dummy_config, conll2003datasetreader_load):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BasePyTorchModel(config=dummy_config,
                             datasets=[conll2003datasetreader_load],
                             # to test passing of arbitrary keyword args to constructor
                             totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_base_pytorch_model(dummy_config_compound_dataset, conll2003datasetreader_load, dummy_dataset_2):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BasePyTorchModel(config=dummy_config_compound_dataset,
                             datasets=[conll2003datasetreader_load, dummy_dataset_2],
                             # to test passing of arbitrary keyword args to constructor
                             totally_arbitrary='arbitrary')
    return model


####################################################################################################
# Keras models
####################################################################################################

# BiLSTM-CRF for NER

@pytest.fixture
def bilstm_crf_model(dummy_config, conll2003datasetreader_load):
    """Returns an instance of BiLSTMCRF initialized with the default configuration and a
    single dataset."""
    model = BiLSTMCRF(config=dummy_config,
                      datasets=[conll2003datasetreader_load],
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_bilstm_crf_model(dummy_config_compound_dataset, conll2003datasetreader_load, dummy_dataset_2):
    """Returns an instance of BiLSTMCRF initialized with the default configuration and a
    compound dataset"""
    model = BiLSTMCRF(config=dummy_config_compound_dataset,
                      datasets=[conll2003datasetreader_load, dummy_dataset_2],
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def bilstm_crf_model_specify(bilstm_crf_model):
    """Returns an instance of BiLSTMCRF initialized with the default configuration file and
    a single specified model."""
    bilstm_crf_model.specify()
    bilstm_crf_model.compile()

    return bilstm_crf_model


@pytest.fixture
def mt_bilstm_crf_model_specify(mt_bilstm_crf_model):
    """Returns an instance of BiLSTMCRF initialized with the default configuration file and
    a single specified model."""
    mt_bilstm_crf_model.specify()
    mt_bilstm_crf_model.compile()

    return mt_bilstm_crf_model


@pytest.fixture
def bilstm_crf_model_embeddings(dummy_config, conll2003datasetreader_load, dummy_embeddings):
    """Returns an instance of BiLSTMCRF initialized with the default configuration file and
    loaded embeddings"""
    model = BiLSTMCRF(config=dummy_config,
                      datasets=[conll2003datasetreader_load],
                      embeddings=dummy_embeddings,
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def bilstm_crf_model_embeddings_specify(bilstm_crf_model_embeddings):
    """Returns an instance of BiLSTMCRF initialized with the default configuration file,
    loaded embeddings and single specified model."""
    bilstm_crf_model_embeddings.specify()

    return bilstm_crf_model_embeddings


@pytest.fixture
def bilstm_crf_model_save(dummy_dir, bilstm_crf_model_specify):
    """Saves a model by calling `bilstm_crf_model_specify.save()` and returns the
    filepath to the saved model."""
    model = bert_for_ner_model_specify

    model_filepath = os.path.join(dummy_dir, constants.KERAS_MODEL_FILENAME)
    weights_filepath = os.path.join(dummy_dir, constants.WEIGHTS_FILENAME)

    bilstm_crf_model_specify.save(model_filepath=model_filepath,
                                  weights_filepath=weights_filepath)

    return model, model_filepath, weights_filepath


@pytest.fixture
def mt_bilstm_crf_model_save(dummy_dir, mt_bilstm_crf_model_specify):
    """Saves a model by calling `single_bert_for_ner_model_specify.save()` and returns the
    filepath to the saved model."""
    model = bert_for_ner_model_specify

    model_filepath = os.path.join(dummy_dir, constants.KERAS_MODEL_FILENAME)
    weights_filepath = os.path.join(dummy_dir, constants.WEIGHTS_FILENAME)

    mt_bilstm_crf_model_specify.save(model_filepath=model_filepath,
                                     weights_filepath=weights_filepath)

    return model, model_filepath, weights_filepath


####################################################################################################
# PyTorch models
####################################################################################################


@pytest.fixture
def bert_tokenizer():
    """Tokenizer for pre-trained BERT model.
    """
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

    return bert_tokenizer


# BertForNER

@pytest.fixture
def bert_for_ner_model(dummy_config, conll2003datasetreader_load):
    """Returns an instance of BertForTokenClassification initialized with the default
    configuration."""
    model = BertForNER(config=dummy_config,
                       datasets=[conll2003datasetreader_load],
                       # to test passing of arbitrary keyword args to constructor
                       totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_bert_for_ner_model(dummy_config_compound_dataset, conll2003datasetreader_load, dummy_dataset_2):
    """Returns an instance of BertForTokenClassification initialized with the default
    configuration."""
    model = BertForNER(config=dummy_config_compound_dataset,
                       datasets=[conll2003datasetreader_load, dummy_dataset_2],
                       # to test passing of arbitrary keyword args to constructor
                       totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def bert_for_ner_model_specify(bert_for_ner_model):
    """Returns an instance of BertForTokenClassification initialized with the default configuration
    file and a single specified model."""
    bert_for_ner_model.specify()

    return bert_for_ner_model


@pytest.fixture
def mt_bert_for_ner_model_specify(mt_bert_for_ner_model):
    """Returns an instance of BertForTokenClassification initialized with the default configuration
    file and a single specified model."""
    mt_bert_for_ner_model.specify()

    return mt_bert_for_ner_model


@pytest.fixture
def bert_for_ner_model_save(dummy_dir, bert_for_ner_model_specify):
    """Saves a model by calling `single_bert_for_ner_model_specify.save()` and returns the
    filepath to the saved model."""
    model = bert_for_ner_model_specify
    model_filepath = os.path.join(dummy_dir, constants.PYTORCH_MODEL_FILENAME)

    bert_for_ner_model_specify.save(model_filepath=model_filepath)

    return model, model_filepath


@pytest.fixture
def mt_bert_for_ner_model_save(dummy_dir, mt_bert_for_ner_model_specify):
    """Saves a model by calling `single_bert_for_ner_model_specify.save()` and returns the
    filepath to the saved model."""
    model = mt_bert_for_ner_model_specify
    model_filepath = os.path.join(dummy_dir, constants.PYTORCH_MODEL_FILENAME)

    mt_bert_for_ner_model_specify.save(model_filepath=model_filepath)

    return model, model_filepath


# BertForJointNERAndRE

@pytest.fixture
def bert_for_joint_ner_and_rc_model(dummy_config, conll2004datasetreader_load):
    """Returns an instance of BertForTokenClassification initialized with the default
    configuration."""
    model = BertForJointNERAndRE(config=dummy_config,
                                 datasets=[conll2004datasetreader_load],
                                 # to test passing of arbitrary keyword args to constructor
                                 totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def bert_for_joint_ner_and_rc_specify(bert_for_joint_ner_and_rc_model):
    """Returns an instance of BertForTokenClassification initialized with the default configuration
    file and a single specified model."""
    bert_for_joint_ner_and_rc_model.specify()

    return bert_for_joint_ner_and_rc_model


@pytest.fixture
def bert_for_joint_ner_and_rc_save(dummy_dir, bert_for_joint_ner_and_rc_specify):
    """Saves a model by calling `single_bert_for_ner_model_specify.save()` and returns the
    filepath to the saved model."""
    model = bert_for_joint_ner_and_rc_specify
    model_filepath = os.path.join(dummy_dir, constants.PYTORCH_MODEL_FILENAME)

    bert_for_joint_ner_and_rc_specify.save(model_filepath=model_filepath)

    return model, model_filepath


####################################################################################################
# Metrics
####################################################################################################


@pytest.fixture
def dummy_metrics(dummy_config,
                  bilstm_crf_model_specify,
                  conll2003datasetreader_load,
                  dummy_training_data,
                  dummy_output_dir):
    """Returns an instance of Metrics.
    """
    metrics = Metrics(config=dummy_config,
                      model_=bilstm_crf_model_specify,
                      training_data=dummy_training_data,
                      idx_to_tag=conll2003datasetreader_load.idx_to_tag,
                      output_dir=dummy_output_dir[0],
                      model_idx=0,
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return metrics


####################################################################################################
# Annotations
####################################################################################################

@pytest.fixture
def blank_annotation():
    """Returns an annotation with no identified entities.
    """
    annotation = {"ents": [],
                  "text": "This is a test with no entities.",
                  "title": ""}
    return annotation


@pytest.fixture
def ched_annotation():
    """Returns an annotation with chemical entities (CHED) identified.
    """
    annotation = {"ents": [{"text": "glucose", "label": "CHED", "start": 0, "end": 0},
                           {"text": "fructose", "label": "CHED", "start": 0, "end": 0}],
                  "text": "glucose and fructose",
                  "title": ""}

    return annotation


@pytest.fixture
def diso_annotation():
    """Returns an annotation with disease entities (DISO) identified.
    """
    annotation = {"ents": [{"text": "cancer", "label": "DISO", "start": 0, "end": 0},
                           {"text": "cystic fibrosis", "label": "DISO", "start": 0, "end": 0}],
                  "text": "cancer and cystic fibrosis",
                  "title": ""}

    return annotation


@pytest.fixture
def livb_annotation():
    """Returns an annotation with species entities (LIVB) identified.
    """
    annotation = {"ents": [{"text": "mouse", "label": "LIVB", "start": 0, "end": 0},
                           {"text": "human", "label": "LIVB", "start": 0, "end": 0}],
                  "text": "mouse and human",
                  "title": ""}

    return annotation


@pytest.fixture
def prge_annotation():
    """Returns an annotation with protein/gene entities (PRGE) identified.
    """
    annotation = {"ents": [{"text": "p53", "label": "PRGE", "start": 0, "end": 0},
                           {"text": "MK2", "label": "PRGE", "start": 0, "end": 0}],
                  "text": "p53 and MK2",
                  "title": ""}

    return annotation


####################################################################################################
# Preprocessing
####################################################################################################


@pytest.fixture
def preprocessor():
    """Returns an instance of a Preprocessor object."""
    return Preprocessor()


@pytest.fixture
def nlp():
    """Returns Sacy NLP model."""
    nlp = spacy.load(constants.SPACY_MODEL)

    return nlp


@pytest.fixture
def nlp_with_biomedical_tokenizer(nlp):
    """Returns an instance of a spaCy's nlp object after replacing the default tokenizer with
    our modified one."""
    nlp.tokenizer = text_utils.biomedical_tokenizer(nlp)

    return nlp
