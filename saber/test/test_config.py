import pytest

from config import Config

# constants for dummy config/CL arguments to perform testing on
PATH_TO_DUMMY_CONFIG = 'saber/test/resources/dummy_config.ini'
DUMMY_PARAMETERS_NO_CLI_ARGS = {
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
'dropout_rate': {'input': 0.3, 'output':0.3, 'recurrent': 0.1},
'batch_size': 1,
'k_folds': 2,
'maximum_number_of_epochs': 10,
'verbose': False,
'debug': False,
'trainable_token_embeddings': False,
'replace_rare_tokens': False}
DUMMY_PARAMETERS_WITH_CLI_ARGS = {
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
'learning_rate': 0.05,
'decay': 0.5,
'gradient_normalization': 1.0,
'dropout_rate': {'input': 0.3, 'output':0.3, 'recurrent': 0.1},
'batch_size': 1,
'k_folds': 2,
'maximum_number_of_epochs': 10,
'verbose': False,
'debug': False,
'trainable_token_embeddings': False,
'replace_rare_tokens': False}
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
    configs = [config_no_cli_args.config, config_with_cli_args.config]
    dummies = [DUMMY_PARAMETERS_NO_CLI_ARGS, DUMMY_PARAMETERS_WITH_CLI_ARGS]
    config_sections = ['mode', 'data', 'model', 'training', 'advanced']

    # Check that all arguments in the config object are as expected. For the
    # special case of lists, first create a string representation (as it would
    # appear in the config file). For the special case of None, we cheat by
    # setting the expected value equal to the actual value. None's occur due to
    # post processing, which we aren't testing here.
    for config, dummy in zip(configs, dummies):
        for section in config_sections:
            for k, v in config[section].items():
                expected = dummy[k]
                # special case of lists
                if isinstance(expected, list):
                    expected = ' '.join(expected)
                # special case of listsdictionaries
                elif isinstance(expected, dict):
                    expected = ', '.join([str(x) for x in expected.values()])
                # special case of None
                elif expected is None or k in DUMMY_COMMAND_LINE_ARGS:
                    expected = v

                assert v == str(expected)

def test_process_parameters_no_command_line_args(config_no_cli_args):
    """Asserts that the parameters are of the expected value/type after a
    call to process_parameters, with NO command line arguments.
    """
    # check that we get the values we expected
    for k, v in DUMMY_PARAMETERS_NO_CLI_ARGS.items():
        assert v == getattr(config_no_cli_args, k)

def test_process_parameters_with_cli_args(config_with_cli_args):
    """Asserts that the parameters are of the expected value/type after a
    call to process_parameters, taking into account command line arguments,
    which take precedence over config arguments.
    """
    # check that we get the values we expected, specifically, check that
    # our command line arguments have overwritten our config arguments
    for k, v in DUMMY_PARAMETERS_WITH_CLI_ARGS.items():
        assert v == getattr(config_with_cli_args, k)
