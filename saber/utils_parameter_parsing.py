import os
import sys
import argparse
import configparser

# TODO: (johngiorgi): use the supported datatypes functions for bools: https://docs.python.org/3.6/library/configparser.html#supported-datatypes
# TODO: (johngiorgi): not clear if the post processing is neccecary

def config_parser(config_filepath):
    """ Returns a parsed config file object.

    Args:
        config_filepath: filepath to .ini config file
    """
    config = configparser.ConfigParser()
    config.read(config_filepath)
    return config

def process_parameters(config, cli_arguments={}):
    """ Load parameters from ini file if specificed.

    Loads parameters from the ini file if specified, taking into account any
    command line arguments, and ensures that each parameter is cast to the
    correct type. Command line arguments take precedence over parameters
    specified in the parameter file.

    Args:
        parameters_filepath: path to ini file containing the parameters

    Returns:
        parameters, a dictionary of parameters (keys) and their values
    """
    parameters = {}

    # parse config
    # mode
    parameters['model_name'] = str(config['mode']['model_name'])
    parameters['train_model'] = bool('True' == config['mode']['train_model'])
    parameters['load_pretrained_model'] = bool('True' == config['mode']['load_pretrained_model'])

    # data
    parameters['dataset_folder'] = str(config['data']['dataset_folder']).split()
    parameters['output_folder'] = str(config['data']['output_folder'])
    parameters['pretrained_model_weights'] = str(config['data']['pretrained_model_weights'])
    parameters['token_pretrained_embedding_filepath'] = str(config['data']['token_pretrained_embedding_filepath'])
    parameters['token_embedding_dimension'] = int(config['data']['token_embedding_dimension'])
    parameters['character_embedding_dimension'] = int(config['data']['character_embedding_dimension'])

    # training
    parameters['optimizer'] = str(config['training']['optimizer'])
    parameters['activation_function'] = str(config['training']['activation_function'])
    parameters['learning_rate'] = float(config['training']['learning_rate'])
    parameters['decay'] = float(config['training']['decay'])
    parameters['gradient_normalization'] = float(config['training']['gradient_normalization'])
    parameters['dropout_rate'] = float(config['training']['dropout_rate'])
    parameters['batch_size'] = int(config['training']['batch_size'])
    parameters['k_folds'] = int(config['training']['k_folds'])
    parameters['maximum_number_of_epochs'] = int(config['training']['maximum_number_of_epochs'])

    # advanced
    parameters['verbose'] = bool('True' == config['advanced']['verbose'])
    parameters['debug'] = bool('True' == config['advanced']['debug'])
    parameters['freeze_token_embeddings'] = bool('True' == config['advanced']['freeze_token_embeddings'])
    parameters['max_char_seq_len'] = int(config['advanced']['max_char_seq_len'])

    # overwrite any parameters in the config if specfied at CL
    for key, value in cli_arguments.items():
        if value is not None:
            parameters[key] = value

    # do any post-processing here
    # replace all whitespace with single space, create list of filepaths
    parameters['dataset_folder'] = [x.strip() for x in parameters['dataset_folder']]
    # lowercase all str arguments (expect directory/file paths)
    parameters['optimizer'] = parameters['optimizer'].strip().lower()
    parameters['activation_function'] = parameters['activation_function'].strip().lower()
    # Do not use gradient normalization if config value is 0
    if parameters['gradient_normalization'] == 0:
        parameters['gradient_normalization'] = None

    return parameters

def parse_arguments():
    """Parse command line (CL) arguments passed with call to Saber.

    Returns:
        a dictionary of parsed CL arguments.
    """
    parser = argparse.ArgumentParser(description='Saber CLI')

    parser.add_argument('--config_filepath',
                        default=os.path.join('.', 'config.ini'),
                        help = '''path to the .ini file containing the
                        parameters. Defaults to './config.ini''')

    parser.add_argument('--activation_function', required=False, type=str, help='')
    parser.add_argument('--batch_size', required=False, type=int, help='')
    parser.add_argument('--character_embedding_dimension', required=False, type=str, help='')
    # parser.add_argument('--character_lstm_hidden_state_dimension', default=argument_default_value, help='')
    # parser.add_argument('--check_for_digits_replaced_with_zeros', default=argument_default_value, help='')
    # parser.add_argument('--check_for_lowercase', default=argument_default_value, help='')
    parser.add_argument('--dataset_folder', required=False, nargs='*', help='')
    parser.add_argument('--debug', required=False, type=bool, help='')
    parser.add_argument('--decay', required=False, type=float, help='')
    parser.add_argument('--dropout_rate', required=False, type=float, help='')
    parser.add_argument('--freeze_token_embeddings', required=False, type=bool, help='')
    parser.add_argument('--gradient_normalization', required=False, type=float, help='')
    parser.add_argument('--k_folds', required=False, type=int, help='')
    parser.add_argument('--learning_rate', required=False, type=float, help='')
    parser.add_argument('--load_pretrained_model', required=False, type=bool, help='')
    # parser.add_argument('--load_only_pretrained_token_embeddings',   default=argument_default_value, help='')
    # parser.add_argument('--load_all_pretrained_token_embeddings',   default=argument_default_value, help='')
    # parser.add_argument('--main_evaluation_mode',   default=argument_default_value, help='')
    parser.add_argument('--maximum_number_of_epochs', required=False, type=int, help='')
    parser.add_argument('--model_name', required=False, type=str, help='')
    # parser.add_argument('--number_of_cpu_threads',   default=argument_default_value, help='')
    # parser.add_argument('--number_of_gpus',   default=argument_default_value, help='')
    parser.add_argument('--optimizer', required=False, type=str, help='')
    parser.add_argument('--output_folder', required=False, type=str, help='')
    # parser.add_argument('--patience', default=argument_default_value, help='')
    # parser.add_argument('--plot_format', default=argument_default_value, help='')
    parser.add_argument('--pretrained_model_weights', required=False, type=str, help='')
    # parser.add_argument('--reload_character_embeddings', default=argument_default_value, help='')
    # parser.add_argument('--reload_character_lstm', default=argument_default_value, help='')
    # parser.add_argument('--reload_crf', default=argument_default_value, help='')
    # parser.add_argument('--reload_feedforward', default=argument_default_value, help='')
    # parser.add_argument('--reload_token_embeddings', default=argument_default_value, help='')
    # parser.add_argument('--reload_token_lstm', default=argument_default_value, help='')
    # parser.add_argument('--remap_unknown_tokens_to_unk', default=argument_default_value, help='')
    # parser.add_argument('--spacylanguage', default=argument_default_value, help='')
    # parser.add_argument('--tagging_format', default=argument_default_value, help='')
    parser.add_argument('--token_embedding_dimension', required=False, type=int, help='')
    # parser.add_argument('--token_lstm_hidden_state_dimension', default=argument_default_value, help='')
    parser.add_argument('--token_pretrained_embedding_filepath', required=False, type=str, help='')
    # parser.add_argument('--tokenizer', default=argument_default_value, help='')
    parser.add_argument('--train_model', required=False, type=bool, help='')
    parser.add_argument('--max_char_seq_len', required=False, type=int, help='')
    # parser.add_argument('--use_character_lstm', default=argument_default_value, help='')
    # parser.add_argument('--use_crf', default=argument_default_value, help='')
    # parser.add_argument('--use_pretrained_model', default=argument_default_value, help='')
    parser.add_argument('--verbose', required=False, action='store_true', help='')

    try:
        cli_arguments = parser.parse_args()
    except:
        parser.print_help()
        sys.ext(0)

    cli_arguments = vars(cli_arguments)
    return cli_arguments
