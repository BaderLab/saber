import numpy
import pytest

from config import Config
from sequence_processor import SequenceProcessor

# constants for dummy dataset/config/word embeddings to perform testing on
PATH_TO_DUMMY_CONFIG = 'saber/test/resources/dummy_config.ini'
PATH_TO_DUMMY_DATASET = 'saber/test/resources/dummy_dataset'
PATH_TO_DUMMY_TOKEN_EMBEDDINGS = 'saber/test/resources/dummy_word_embeddings/dummy_word_embeddings.txt'
DUMMY_TRAIN_SENT_NUM = 2
DUMMY_TEST_SENT_NUM = 1
DUMMY_TAG_TYPE_COUNT = 5
# embedding matrix shape is num word types x dimension of embeddings
DUMMY_EMBEDDINGS_MATRIX_SHAPE = (25, 2)

# TODO (johngiorgi): add some kind of test that accounts for the error thrown
# when we try to load token embedding before loading a dataset.

@pytest.fixture
def dummy_config_single_ds():
    """Returns an instance of a configparser object after parsing the dummy
    config file. """
    # create the config object, taking into account the CLI args
    dummy_config_single_ds = Config(PATH_TO_DUMMY_CONFIG)

    return dummy_config_single_ds

@pytest.fixture
def dummy_config_compound_ds():
    """Returns an instance of a configparser object after parsing the dummy
    config file. """
    # create the config object, taking into account the CLI args
    compound_dataset = [PATH_TO_DUMMY_DATASET, PATH_TO_DUMMY_DATASET]
    cli_arguments = {'dataset_folder': compound_dataset}
    dummy_config_compound_ds = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config_compound_ds.process_parameters(cli_arguments)

    return dummy_config_compound_ds

@pytest.fixture
def sp_no_ds_no_embed(dummy_config_single_ds):
    """Returns an instance of SequenceProcessor initialized with the
    default configuration file and no loaded dataset. """
    sp_no_ds_no_embed = SequenceProcessor(config=dummy_config_single_ds)

    return sp_no_ds_no_embed

@pytest.fixture
def sp_sing_ds_no_embed(dummy_config_single_ds):
    """Returns an instance of SequenceProcessor initialized with the
    default configuration file and a single loaded dataset."""
    sp_sing_ds_no_embed = SequenceProcessor(config=dummy_config_single_ds)
    sp_sing_ds_no_embed.load_dataset()

    return sp_sing_ds_no_embed

@pytest.fixture
def sp_compound_ds_no_embed(dummy_config_compound_ds):
    """Returns an instance of SequenceProcessor initialized with the
    default configuration file and a compound loaded dataset. The compound
    dataset is just two copies of the dataset, this makes writing tests
    much simpler."""
    sp_compound_ds_no_embed = SequenceProcessor(config=dummy_config_compound_ds)
    sp_compound_ds_no_embed.load_dataset()

    return sp_compound_ds_no_embed

@pytest.fixture
def sp_single_ds_no_embed_with_model(dummy_config_single_ds):
    """Returns an instance of SequenceProcessor initialized with the
    default configuration file, a single loaded dataset and a keras model."""
    sp_single_ds_no_embed_with_model = SequenceProcessor(config=dummy_config_single_ds)
    sp_single_ds_no_embed_with_model.load_dataset()
    sp_single_ds_no_embed_with_model.create_model()

    return sp_single_ds_no_embed_with_model

def test_attributes_after_initilization_of_model(sp_no_ds_no_embed):
    """Asserts instance attributes are initialized correctly when sequence
    model is initialized (and before dataset is loaded)."""
    # check value/type
    assert sp_no_ds_no_embed.config.activation_function == 'relu'
    assert sp_no_ds_no_embed.config.batch_size == 1
    assert sp_no_ds_no_embed.config.character_embedding_dimension == 30
    assert sp_no_ds_no_embed.config.dataset_folder == [PATH_TO_DUMMY_DATASET]
    assert sp_no_ds_no_embed.config.debug == False
    assert sp_no_ds_no_embed.config.dropout_rate == {'input': 0.3, 'output':0.3, 'recurrent': 0.1}
    assert sp_no_ds_no_embed.config.trainable_token_embeddings == False
    assert sp_no_ds_no_embed.config.gradient_normalization == None
    assert sp_no_ds_no_embed.config.k_folds == 2
    assert sp_no_ds_no_embed.config.learning_rate == 0.01
    assert sp_no_ds_no_embed.config.decay == 0.05
    assert sp_no_ds_no_embed.config.load_pretrained_model == False
    assert sp_no_ds_no_embed.config.maximum_number_of_epochs == 10
    assert sp_no_ds_no_embed.config.model_name == 'MT-LSTM-CRF'
    assert sp_no_ds_no_embed.config.optimizer == 'sgd'
    assert sp_no_ds_no_embed.config.output_folder == '../output'
    assert sp_no_ds_no_embed.config.pretrained_model_weights == ''
    assert sp_no_ds_no_embed.config.token_embedding_dimension == 200
    assert sp_no_ds_no_embed.config.token_pretrained_embedding_filepath == PATH_TO_DUMMY_TOKEN_EMBEDDINGS
    assert sp_no_ds_no_embed.config.train_model == True
    assert sp_no_ds_no_embed.config.verbose == False

    assert sp_no_ds_no_embed.ds == []
    assert sp_no_ds_no_embed.token_embedding_matrix == None
    assert sp_no_ds_no_embed.model == None

def test_token_embeddings_load(sp_sing_ds_no_embed,
                               sp_compound_ds_no_embed):
    """Asserts that pre-trained token embeddings are loaded correctly when
    SequenceProcessor.load_embeddings() is called"""
    # load embeddings for each model
    sp_sing_ds_no_embed.load_embeddings()
    sp_compound_ds_no_embed.load_embeddings()

    # check type
    assert isinstance(sp_sing_ds_no_embed.token_embedding_matrix, numpy.ndarray)
    assert isinstance(sp_compound_ds_no_embed.token_embedding_matrix, numpy.ndarray)
    # check value
    assert sp_sing_ds_no_embed.token_embedding_matrix.shape == DUMMY_EMBEDDINGS_MATRIX_SHAPE
    assert sp_compound_ds_no_embed.token_embedding_matrix.shape == DUMMY_EMBEDDINGS_MATRIX_SHAPE

def test_X_input_sequences_after_loading_single_dataset(sp_sing_ds_no_embed):
    """Asserts X (input) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for single
    datasets."""
    # shortens assert statments
    ds = sp_sing_ds_no_embed.ds[0]
    # check type
    assert isinstance(ds.train_word_idx_seq, numpy.ndarray)
    # check shape
    assert ds.train_word_idx_seq.shape[0] == DUMMY_TRAIN_SENT_NUM

def test_y_output_sequences_after_loading_single_dataset(sp_sing_ds_no_embed):
    """Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for single
    datasets."""
    # shortens assert statments
    ds = sp_sing_ds_no_embed.ds[0]
    model = sp_sing_ds_no_embed
    # check type
    assert isinstance(ds.train_tag_idx_seq, numpy.ndarray)
    # check value
    assert ds.train_tag_idx_seq.shape[0] == DUMMY_TRAIN_SENT_NUM
    assert ds.train_tag_idx_seq.shape[-1] == DUMMY_TAG_TYPE_COUNT

def test_agreement_between_model_and_single_dataset(sp_sing_ds_no_embed):
    """Asserts that the attributes common to SequenceProcessor and
    Dataset are the same for single datasets."""
    # shortens assert statments
    ds = sp_sing_ds_no_embed.ds[0]
    model = sp_sing_ds_no_embed

    assert model.config.dataset_folder[0] == ds.filepath

def test_X_input_sequences_after_loading_compound_dataset(sp_compound_ds_no_embed):
    """Asserts X (input) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for compound
    datasets."""
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for ds in sp_compound_ds_no_embed.ds:
        # check type
        assert isinstance(ds.train_word_idx_seq, numpy.ndarray)
        # check shape
        assert ds.train_word_idx_seq.shape[0] == DUMMY_TRAIN_SENT_NUM

def test_y_output_sequences_after_loading_compound_dataset(sp_compound_ds_no_embed):
    """Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for compound
    datasets."""
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for ds in sp_compound_ds_no_embed.ds:
        assert isinstance(ds.train_tag_idx_seq, numpy.ndarray)
        # check value
        assert ds.train_tag_idx_seq.shape[0] == DUMMY_TRAIN_SENT_NUM
        assert ds.train_tag_idx_seq.shape[-1] == DUMMY_TAG_TYPE_COUNT

def test_agreement_between_model_and_compound_dataset(sp_compound_ds_no_embed):
    """Asserts that the attributes common to SequenceProcessor and Dataset are
    the same for compound datasets."""
    # shortens assert statments
    model = sp_compound_ds_no_embed
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for i, ds in enumerate(model.ds):
        assert model.config.dataset_folder[i] == ds.filepath

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
