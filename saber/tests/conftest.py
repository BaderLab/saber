"""A collection of PyTest fixtures used for Sabers unit test (saber/tests/test_*.py)
"""
import pytest
from pytorch_pretrained_bert import BertTokenizer

from ..config import Config
from ..dataset import Dataset
from ..embeddings import Embeddings
from ..metrics import Metrics
from ..models.base_model import BaseModel
from ..models.base_model import BaseKerasModel
from ..models.base_model import BasePyTorchModel
from ..models.bert_for_ner import BertForNER
from ..models.bilstm_crf import BiLSTMCRF
from ..preprocessor import Preprocessor
from ..saber import Saber
from ..utils import data_utils, model_utils, text_utils
from .resources.constants import *
from .. import constants
import spacy
import os


# generic

@pytest.fixture(scope='session')
def dummy_dir(tmpdir_factory):
    """Returns the path to a temporary directory.
    """
    dummy_dir = tmpdir_factory.mktemp('dummy_dir')
    return dummy_dir.strpath

# config


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
    compound_dataset = [PATH_TO_DUMMY_DATASET_1, PATH_TO_DUMMY_DATASET_2]
    cli_arguments = {'dataset_folder': compound_dataset}
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config.harmonize_args(cli_arguments)

    return dummy_config

# dataset


@pytest.fixture
def empty_dummy_dataset():
    """Returns an empty single dummy Dataset instance.
    """
    # Don't replace rare tokens for the sake of testing
    return Dataset(dataset_folder=PATH_TO_DUMMY_DATASET_1, replace_rare_tokens=False,
                   # to test passing of arbitrary keyword args to constructor
                   totally_arbitrary='arbitrary')


@pytest.fixture
def dummy_dataset_1():
    """Returns a single dummy Dataset instance after calling `Dataset.load()`.
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(dataset_folder=PATH_TO_DUMMY_DATASET_1, replace_rare_tokens=False)
    dataset.load()

    return dataset


@pytest.fixture
def dummy_dataset_2():
    """Returns a single dummy Dataset instance after calling `Dataset.load()`.
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(dataset_folder=PATH_TO_DUMMY_DATASET_2, replace_rare_tokens=False)
    dataset.load()

    return dataset


@pytest.fixture
def dummy_compound_dataset(dummy_config):
    """
    """
    dummy_config.dataset_folder = [PATH_TO_DUMMY_DATASET_1, PATH_TO_DUMMY_DATASET_2]
    dummy_config.replace_rare_tokens = False
    dataset = data_utils.load_compound_dataset(dummy_config)

    return dataset


@pytest.fixture(scope='session')
def dummy_dataset_paths_all(tmpdir_factory):
    """Creates and returns the path to a temporary dataset folder, and train, valid, test files.
    """
    # create a dummy dataset folder
    dummy_dir = tmpdir_factory.mktemp('dummy_dataset')
    # create train, valid and train partitions in this folder
    train_file = dummy_dir.join('train.tsv')
    train_file.write('arbitrary') # need to write content or else the file wont exist
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


# embeddings

@pytest.fixture
def dummy_embeddings(dummy_dataset_1):
    """Returns an instance of an `Embeddings()` object AFTER the `.load()` method is called.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS,
                            token_map=dummy_dataset_1.idx_to_tag)
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
    embedding_matrix, type_to_idx = embeddings._prepare_embedding_matrix(embedding_idx, load_all=False)
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
    embedding_matrix, type_to_idx = embeddings._prepare_embedding_matrix(embedding_idx, load_all=True)
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

# saber


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
    saber.load_dataset(directory=PATH_TO_DUMMY_DATASET_1)

    return saber


@pytest.fixture
def saber_single_dataset_embeddings(dummy_config):
    """Returns instance of `Saber` initialized with the dummy config file, a single dataset and
    embeddings.
    """
    saber = Saber(config=dummy_config)
    saber.load_dataset(directory=PATH_TO_DUMMY_DATASET_1)
    saber.load_embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, binary=False)

    return saber


@pytest.fixture
def saber_single_dataset_model(dummy_config):
    """Returns an instance of `Saber` initialized with the dummy config file, a single dataset
    a Keras model."""
    saber = Saber(config=dummy_config)
    saber.load_dataset(directory=PATH_TO_DUMMY_DATASET_1)
    saber.build()

    return saber


@pytest.fixture
def saber_compound_dataset(dummy_config_compound_dataset):
    """Returns an instance of `Saber` initialized with the dummy config file and a compound dataset.
    The compound dataset is just two copies of the dataset, this makes writing tests much
    simpler.
    """
    compound_dataset = [PATH_TO_DUMMY_DATASET_1, PATH_TO_DUMMY_DATASET_1]
    saber = Saber(config=dummy_config_compound_dataset)
    saber.load_dataset(directory=compound_dataset)

    return saber


@pytest.fixture
def saber_compound_dataset_model(dummy_config_compound_dataset):
    """Returns an instance of `Saber` initialized with the dummy config file, a single dataset
    a Keras model."""
    saber = Saber(config=dummy_config_compound_dataset)
    saber.load_dataset(directory=[PATH_TO_DUMMY_DATASET_1, PATH_TO_DUMMY_DATASET_2])
    saber.build()

    return saber


@pytest.fixture
def saber_saved_model(dummy_dir, dummy_config):
    """Returns a tuple containing an instance of a `Saber` object after `save()` was called and
    its models wiped (`saber.models = []`), the models and datasets it was saved with, and the
    directory where the model was saved.
    """
    saber = Saber(config=dummy_config)
    saber.load_dataset(directory=PATH_TO_DUMMY_DATASET_1)
    saber.build()

    model, dataset = saber.models[-1], saber.datasets[-1]

    directory = saber.save(directory=dummy_dir)

    saber.models = []

    return saber, model, dataset, directory

# model training


@pytest.fixture
def dummy_output_dir(tmpdir, dummy_config):
    """Returns list of output directories."""
    # make sure top-level directory is the pytest tmpdir
    dummy_config.output_folder = tmpdir.strpath
    output_dirs = model_utils.prepare_output_directory(dummy_config)

    return output_dirs


@pytest.fixture
def dummy_training_data(dummy_dataset_1):
    """Returns training data from `dummy_dataset_1`.
    """
    training_data = {'x_train': [dummy_dataset_1.idx_seq['train']['word'],
                                 dummy_dataset_1.idx_seq['train']['char']],
                     'x_valid': None,
                     'x_test': None,
                     'y_train': dummy_dataset_1.idx_seq['train']['tag'],
                     'y_valid': None,
                     'y_test': None,
                     }

    return training_data

# Keras models


@pytest.fixture
def bilstm_crf_model(dummy_config, dummy_dataset_1):
    """Returns an instance of BiLSTMCRF initialized with the default configuration and a
    single dataset."""
    model = BiLSTMCRF(config=dummy_config,
                      datasets=[dummy_dataset_1],
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_bilstm_crf_model(dummy_config, dummy_dataset_1, dummy_dataset_2):
    """Returns an instance of BiLSTMCRF initialized with the default configuration and a
    compound dataset"""
    model = BiLSTMCRF(config=dummy_config,
                      datasets=[dummy_dataset_1, dummy_dataset_2],
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def bilstm_crf_model_specify(bilstm_crf_model):
    """Returns an instance of BiLSTMCRF initialized with the default configuration file and
    a single specified model."""
    bilstm_crf_model.specify()

    return bilstm_crf_model


@pytest.fixture
def mt_bilstm_crf_model_specify(mt_bilstm_crf_model):
    """Returns an instance of BiLSTMCRF initialized with the default configuration file and
    a single specified model."""
    mt_bilstm_crf_model.specify()

    return mt_bilstm_crf_model


@pytest.fixture
def bilstm_crf_model_embeddings(dummy_config, dummy_dataset_1, dummy_embeddings):
    """Returns an instance of BiLSTMCRF initialized with the default configuration file and
    loaded embeddings"""
    model = BiLSTMCRF(config=dummy_config,
                      datasets=[dummy_dataset_1],
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
def base_model(dummy_config, dummy_dataset_1):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BaseModel(config=dummy_config,
                      datasets=[dummy_dataset_1],
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_base_model(dummy_config, dummy_dataset_1, dummy_dataset_2):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BaseModel(config=dummy_config,
                      datasets=[dummy_dataset_1, dummy_dataset_2],
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def base_keras_model(dummy_config, dummy_dataset_1):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BaseKerasModel(config=dummy_config,
                           datasets=[dummy_dataset_1],
                           # to test passing of arbitrary keyword args to constructor
                           totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_base_keras_model(dummy_config, dummy_dataset_1, dummy_dataset_2):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BaseKerasModel(config=dummy_config,
                           datasets=[dummy_dataset_1, dummy_dataset_2],
                           # to test passing of arbitrary keyword args to constructor
                           totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def base_keras_model_embeddings(dummy_config, dummy_dataset_1, dummy_embeddings):
    """Returns an instance of BiLSTMCRF initialized with the default configuration file and
    loaded embeddings"""
    model = BaseKerasModel(config=dummy_config,
                           datasets=[dummy_dataset_1],
                           embeddings=dummy_embeddings,
                           # to test passing of arbitrary keyword args to constructor
                           totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def base_pytorch_model(dummy_config, dummy_dataset_1):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BasePyTorchModel(config=dummy_config,
                             datasets=[dummy_dataset_1],
                             # to test passing of arbitrary keyword args to constructor
                             totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_base_pytorch_model(dummy_config, dummy_dataset_1, dummy_dataset_2):
    """Returns an instance of BiLSTMCRF initialized with the default configuration."""
    model = BasePyTorchModel(config=dummy_config,
                             datasets=[dummy_dataset_1, dummy_dataset_2],
                             # to test passing of arbitrary keyword args to constructor
                             totally_arbitrary='arbitrary')
    return model

# BERT models


@pytest.fixture
def bert_tokenizer():
    """Tokenizer for pre-trained BERT model.
    """
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    return bert_tokenizer


@pytest.fixture
def bert_for_ner_model(dummy_config, dummy_dataset_1):
    """Returns an instance of BertForTokenClassification initialized with the default
    configuration."""
    model = BertForNER(config=dummy_config,
                       datasets=[dummy_dataset_1],
                       # to test passing of arbitrary keyword args to constructor
                       totally_arbitrary='arbitrary')
    return model


@pytest.fixture
def mt_bert_for_ner_model(dummy_config, dummy_dataset_1, dummy_dataset_2):
    """Returns an instance of BertForTokenClassification initialized with the default
    configuration."""
    model = BertForNER(config=dummy_config,
                       datasets=[dummy_dataset_1, dummy_dataset_2],
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

# metrics


@pytest.fixture
def dummy_metrics(dummy_config, dummy_dataset_1, dummy_training_data, dummy_output_dir,
                  base_keras_model):
    """Returns an instance of Metrics.
    """
    metrics = Metrics(config=dummy_config,
                      model_=base_keras_model,
                      training_data=dummy_training_data,
                      idx_to_tag=dummy_dataset_1.idx_to_tag,
                      output_dir=dummy_output_dir,
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return metrics

# annotations


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

# preprocessing


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
