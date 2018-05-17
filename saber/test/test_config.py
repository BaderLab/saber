import pytest

from config import Config

# constants for dummy config/CL arguments to perform testing on
PATH_TO_DUMMY_CONFIG = 'saber/test/resources/dummy_config.ini'
DUMMY_PARAMETERS_NO_COMMAND_LINE_ARGS = {
'model_name': 'MT-LSTM-CRF',
'train_model': True,
'load_pretrained_model': False,
'dataset_folder': ['saber/test/resources/dummy_dataset'],
'output_folder': '../output',
'pretrained_model_weights': '',
'token_pretrained_embedding_filepath': 'saber/test/resources/dummy_word_embeddings/dummy_word_embeddings.txt',
'token_embedding_dimension': 200,
'character_embedding_dimension': 30,
'optimizer': 'sgd',
'activation_function': 'relu',
'learning_rate': 0.01,
'decay': 0.05,
'gradient_normalization': None,
'dropout_rate': 0.3,
'batch_size': 1,
'k_folds': 2,
'maximum_number_of_epochs': 10,
'verbose': False,
'debug': False,
'freeze_token_embeddings': True}
DUMMY_PARAMETERS_WITH_COMMAND_LINE_ARGS = {
'model_name': 'MT-LSTM-CRF',
'train_model': True,
'load_pretrained_model': False,
'dataset_folder': ['saber/test/resources/dummy_dataset'],
'output_folder': '../output', 'pretrained_model_weights': '',
'token_pretrained_embedding_filepath': 'saber/test/resources/dummy_word_embeddings/dummy_word_embeddings.txt',
'token_embedding_dimension': 200,
'character_embedding_dimension': 30,
'optimizer': 'sgd',
'activation_function': 'relu',
'learning_rate': 0.05,
'decay': 0.5,
'gradient_normalization': 1.0,
'dropout_rate': 0.3,
'batch_size': 1,
'k_folds': 2,
'maximum_number_of_epochs': 10,
'verbose': False,
'debug': False,
'freeze_token_embeddings': True}
DUMMY_COMMAND_LINE_ARGS = {
'gradient_normalization': 1.0,
'learning_rate':0.05,
'decay':0.5}

@pytest.fixture
def config_no_cli_args():
    """Returns an instance of a Config object after parsing the dummy
    config file with no command line interface (CLI) args."""
    # parse the dummy config
    config_no_cli_args = Config(config_filepath=PATH_TO_DUMMY_CONFIG,
                                cli=False)

    return config_no_cli_args

@pytest.fixture
def config_with_cli_args():
    """Returns an instance of a Config object after parsing the dummy
    config file with command line interface (CLI) args."""
    # parse the dummy config, leave cli false and instead pass the command
    # line args through Config.process_parameters
    config_with_cli_args = Config(config_filepath=PATH_TO_DUMMY_CONFIG,
                                  cli=False)
    config_with_cli_args.process_parameters(DUMMY_COMMAND_LINE_ARGS)

    return config_with_cli_args

def test_parse_config_args_no_cli_args(config_no_cli_args, config_with_cli_args):
    """Asserts the Config.config object contains the expected values after
    call to Config.parse_config_args(), with and without CLI args."""
    dummy_config_no_cli_args = config_no_cli_args.config
    dummy_config_with_cli_args = config_with_cli_args.config

    assert dummy_config_no_cli_args['mode']['model_name'] == \
        dummy_config_with_cli_args['mode']['model_name'] == 'MT-LSTM-CRF'
    assert dummy_config_no_cli_args['mode']['train_model'] == \
        dummy_config_with_cli_args['mode']['train_model'] == 'True'
    assert dummy_config_no_cli_args['mode']['load_pretrained_model'] == \
        dummy_config_with_cli_args['mode']['load_pretrained_model'] == 'False'

    assert dummy_config_no_cli_args['data']['dataset_folder'] == \
        dummy_config_with_cli_args['data']['dataset_folder'] == 'saber/test/resources/dummy_dataset'
    assert dummy_config_no_cli_args['data']['output_folder'] == \
        dummy_config_with_cli_args['data']['output_folder'] == '../output'
    assert dummy_config_no_cli_args['data']['pretrained_model_weights'] == \
        dummy_config_with_cli_args['data']['pretrained_model_weights'] == ''
    assert dummy_config_no_cli_args['data']['token_pretrained_embedding_filepath'] == \
        dummy_config_with_cli_args['data']['token_pretrained_embedding_filepath'] == \
        'saber/test/resources/dummy_word_embeddings/dummy_word_embeddings.txt'

    assert dummy_config_no_cli_args['training']['optimizer'] == \
        dummy_config_with_cli_args['training']['optimizer'] == 'sgd'
    assert dummy_config_no_cli_args['training']['activation_function'] == \
        dummy_config_with_cli_args['training']['activation_function'] == 'relu'
    assert dummy_config_no_cli_args['training']['learning_rate'] == \
        dummy_config_with_cli_args['training']['learning_rate'] == '0.01'
    assert dummy_config_no_cli_args['training']['decay'] == \
        dummy_config_with_cli_args['training']['decay'] == '0.05'
    assert dummy_config_no_cli_args['training']['gradient_normalization'] == \
        dummy_config_with_cli_args['training']['gradient_normalization'] == '0.0'
    assert dummy_config_no_cli_args['training']['dropout_rate'] == \
        dummy_config_with_cli_args['training']['dropout_rate'] == '0.3'
    assert dummy_config_no_cli_args['training']['batch_size'] == \
        dummy_config_with_cli_args['training']['batch_size'] == '1'
    assert dummy_config_no_cli_args['training']['k_folds'] == \
        dummy_config_with_cli_args['training']['k_folds'] == '2'
    assert dummy_config_no_cli_args['training']['maximum_number_of_epochs'] == \
        dummy_config_with_cli_args['training']['maximum_number_of_epochs'] == '10'

    assert dummy_config_no_cli_args['advanced']['debug'] == \
        dummy_config_with_cli_args['advanced']['debug'] == 'False'
    assert dummy_config_no_cli_args['advanced']['freeze_token_embeddings'] == \
        dummy_config_with_cli_args['advanced']['freeze_token_embeddings'] == 'True'
    assert dummy_config_no_cli_args['advanced']['verbose'] == \
        dummy_config_with_cli_args['advanced']['verbose'] == 'False'

def test_process_parameters_no_command_line_args(config_no_cli_args):
    """Asserts that the parameters are of the expected value/type after a
    call to process_parameters, with NO command line arguments.
    """
    # check that we get the values we expected
    for k, v in DUMMY_PARAMETERS_NO_COMMAND_LINE_ARGS.items():
        assert v == getattr(config_no_cli_args, k)

def test_process_parameters_with_cli_args(config_with_cli_args):
    """Asserts that the parameters are of the expected value/type after a
    call to process_parameters, taking into account command line arguments,
    which take precedence over config arguments.
    """
    # check that we get the values we expected, specifically, check that
    # our command line arguments have overwritten our config arguments
    for k, v in DUMMY_PARAMETERS_WITH_COMMAND_LINE_ARGS.items():
        assert v == getattr(config_with_cli_args, k)
