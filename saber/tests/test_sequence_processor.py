# TODO (johngiorgi): Need to increase coverage to include cases where
# some combination of train, valid and test partitions are provided. Currently,
# only the case where a single partition (train.*) is provided is covered.

from os.path import abspath
import pytest

import numpy

from ..config import Config
from ..sequence_processor import SequenceProcessor

# constants for dummy dataset/config/word embeddings to perform testing on
PATH_TO_DUMMY_CONFIG = abspath('saber/tests/resources/dummy_config.ini')
PATH_TO_DUMMY_DATASET = abspath('saber/tests/resources/dummy_dataset_1')
PATH_TO_DUMMY_TOKEN_EMBEDDINGS = abspath(('saber/tests/resources/dummy_word_embeddings/'
                                          'dummy_word_embeddings.txt'))
DUMMY_TRAIN_SENT_NUM = 2
DUMMY_TEST_SENT_NUM = 1
DUMMY_TAG_TYPE_COUNT = 5
# embedding matrix shape is num word types x dimension of embeddings
DUMMY_EMBEDDINGS_MATRIX_SHAPE = (25, 2)

# TODO (johngiorgi): add some kind of test that accounts for the error thrown
# when we try to load token embedding before loading a dataset.

@pytest.fixture
def dummy_config_single_ds():
    """Returns an instance of a configparser object after parsing the dummy config file. Ensures
    that `replace_rare_tokens` argument is False."""
    cli_arguments = {'replace_rare_tokens': False}
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config._process_args(cli_arguments)

    return dummy_config

@pytest.fixture
def dummy_config_compound_ds():
    """Returns an instance of a configparser object after parsing the dummy
    config file. """
    # create the config object, taking into account the CLI args
    compound_dataset = [PATH_TO_DUMMY_DATASET, PATH_TO_DUMMY_DATASET]
    cli_arguments = {'dataset_folder': compound_dataset, 'replace_rare_tokens': False}
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config._process_args(cli_arguments)

    return dummy_config

@pytest.fixture
def sp_no_ds_no_embed(dummy_config_single_ds):
    """Returns an instance of SequenceProcessor initialized with the default dummy config file and
    no loaded dataset. """
    return SequenceProcessor(config=dummy_config_single_ds)

@pytest.fixture
def sp_sing_ds_no_embed(dummy_config_single_ds):
    """Returns an instance of SequenceProcessor initialized with the dummy config file and a single
    loaded dataset."""
    sequence_processor = SequenceProcessor(config=dummy_config_single_ds)
    sequence_processor.load_dataset()

    return sequence_processor

@pytest.fixture
def sp_compound_ds_no_embed(dummy_config_compound_ds):
    """Returns an instance of SequenceProcessor initialized with the dummy config file and a
    compound loaded dataset. The compound dataset is just two copies of the dataset, this makes
    writing tests much simpler."""
    sequence_processor = SequenceProcessor(config=dummy_config_compound_ds)
    sequence_processor.load_dataset()

    return sequence_processor

@pytest.fixture
def sp_single_ds_no_embed_with_model(dummy_config_single_ds):
    """Returns an instance of SequenceProcessor initialized with the dummy config file, a single
    loaded dataset and a keras model."""
    sequence_processor = SequenceProcessor(config=dummy_config_single_ds)
    sequence_processor.load_dataset()
    sequence_processor.create_model()

    return sequence_processor

def test_attributes_after_initilization_of_model(sp_no_ds_no_embed, dummy_config_single_ds):
    """Asserts instance attributes are initialized correctly when sequence model is initialized
    (and before dataset is loaded)."""
    # SequenceProcessor object attributes
    assert sp_no_ds_no_embed.config is dummy_config_single_ds
    assert sp_no_ds_no_embed.ds == []
    assert sp_no_ds_no_embed.token_embedding_matrix is None
    assert sp_no_ds_no_embed.model is None

    # Attributes of Config object tied to SequenceProcessor instance
    assert sp_no_ds_no_embed.config.activation == 'relu'
    assert sp_no_ds_no_embed.config.batch_size == 32
    assert sp_no_ds_no_embed.config.char_embed_dim == 30
    assert sp_no_ds_no_embed.config.criteria == 'exact'
    assert sp_no_ds_no_embed.config.dataset_folder == [PATH_TO_DUMMY_DATASET]
    assert not sp_no_ds_no_embed.config.debug
    assert sp_no_ds_no_embed.config.decay == 0.0
    assert sp_no_ds_no_embed.config.dropout_rate == {'input': 0.3, 'output':0.3, 'recurrent': 0.1}
    assert not sp_no_ds_no_embed.config.fine_tune_word_embeddings
    assert sp_no_ds_no_embed.config.grad_norm == 1.0
    assert sp_no_ds_no_embed.config.k_folds == 2
    assert sp_no_ds_no_embed.config.learning_rate == 0.0
    assert sp_no_ds_no_embed.config.epochs == 50
    assert sp_no_ds_no_embed.config.model_name == 'mt-lstm-crf'
    assert sp_no_ds_no_embed.config.optimizer == 'nadam'
    assert sp_no_ds_no_embed.config.output_folder == abspath('../output')
    assert sp_no_ds_no_embed.config.pretrained_model_weights is ''
    assert not sp_no_ds_no_embed.config.replace_rare_tokens
    assert sp_no_ds_no_embed.config.word_embed_dim == 200
    assert sp_no_ds_no_embed.config.pretrained_embeddings == PATH_TO_DUMMY_TOKEN_EMBEDDINGS
    assert sp_no_ds_no_embed.config.train_model
    assert not sp_no_ds_no_embed.config.verbose

def test_token_embeddings_load(sp_sing_ds_no_embed, sp_compound_ds_no_embed):
    """Asserts that pre-trained token embeddings are loaded correctly when
    SequenceProcessor.load_embeddings() is called"""
    # load embeddings for each model
    sp_sing_ds_no_embed.load_embeddings(binary=False)
    sp_compound_ds_no_embed.load_embeddings(binary=False)

    # check type
    assert isinstance(sp_sing_ds_no_embed.token_embedding_matrix, numpy.ndarray)
    assert isinstance(sp_compound_ds_no_embed.token_embedding_matrix, numpy.ndarray)
    # check shape
    assert sp_sing_ds_no_embed.token_embedding_matrix.shape == DUMMY_EMBEDDINGS_MATRIX_SHAPE
    assert sp_compound_ds_no_embed.token_embedding_matrix.shape == DUMMY_EMBEDDINGS_MATRIX_SHAPE

def test_X_input_sequences_after_loading_single_dataset(sp_sing_ds_no_embed):
    """Asserts X (input) data partition attribute is initialized correctly when sequence model is
    initialized (and after dataset is loaded) for single datasets."""
    # shortens assert statments
    ds = sp_sing_ds_no_embed.ds[0]
    # check type
    assert isinstance(ds.idx_seq['train']['word'], numpy.ndarray)
    # check shape
    assert ds.idx_seq['train']['word'].shape[0] == DUMMY_TRAIN_SENT_NUM

def test_y_output_sequences_after_loading_single_dataset(sp_sing_ds_no_embed):
    """Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for single
    datasets."""
    # shortens assert statments
    ds = sp_sing_ds_no_embed.ds[0]
    model = sp_sing_ds_no_embed
    # check type
    assert isinstance(ds.idx_seq['train']['tag'], numpy.ndarray)
    # check value
    assert ds.idx_seq['train']['tag'].shape[0] == DUMMY_TRAIN_SENT_NUM
    assert ds.idx_seq['train']['tag'].shape[-1] == DUMMY_TAG_TYPE_COUNT

def test_X_input_sequences_after_loading_compound_dataset(sp_compound_ds_no_embed):
    """Asserts X (input) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for compound
    datasets."""
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for ds in sp_compound_ds_no_embed.ds:
        # check type
        assert isinstance(ds.idx_seq['train']['word'], numpy.ndarray)
        # check shape
        assert ds.idx_seq['train']['word'].shape[0] == DUMMY_TRAIN_SENT_NUM

def test_y_output_sequences_after_loading_compound_dataset(sp_compound_ds_no_embed):
    """Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for compound
    datasets."""
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for ds in sp_compound_ds_no_embed.ds:
        assert isinstance(ds.idx_seq['train']['tag'], numpy.ndarray)
        # check value
        assert ds.idx_seq['train']['tag'].shape[0] == DUMMY_TRAIN_SENT_NUM
        assert ds.idx_seq['train']['tag'].shape[-1] == DUMMY_TAG_TYPE_COUNT

def test_predict(sp_single_ds_no_embed_with_model):
    """Asserts that call to SequenceProcessor.predict() returns the expected
    results."""
    # a simple, single-sentence test
    simple_text = "This is a simple test"
    simple_annotation = {'text': simple_text, 'ents': [], 'title': None}
    # a simple, multi-sentence test
    multi_sentence_text = "This is a simple text. With multiple sentences"
    multi_sentence_annotation = {'text': multi_sentence_text, 'ents': [], 'title': None}

    simple_prediction = sp_single_ds_no_embed_with_model.annotate(simple_text)
    multi_sentence_prediction = sp_single_ds_no_embed_with_model.annotate(multi_sentence_text)
    # wipe the predicted entities as these are stochastic.
    simple_prediction['ents'] = []
    multi_sentence_prediction['ents'] = []

    assert simple_prediction == simple_annotation
    assert multi_sentence_prediction == multi_sentence_annotation

def test_predict_blank_or_invalid(sp_single_ds_no_embed_with_model):
    """Asserts that call to SequenceProcessor.predict() raises an assertion
    error when a falsy text argument is passed."""
    # one test for each falsy type
    blank_text_test = ""
    none_test = None
    empty_list_test = []
    false_bool_test = False

    with pytest.raises(ValueError):
        sp_single_ds_no_embed_with_model.annotate(blank_text_test)
    with pytest.raises(ValueError):
        sp_single_ds_no_embed_with_model.annotate(none_test)
    with pytest.raises(ValueError):
        sp_single_ds_no_embed_with_model.annotate(empty_list_test)
    with pytest.raises(ValueError):
        sp_single_ds_no_embed_with_model.annotate(false_bool_test)
