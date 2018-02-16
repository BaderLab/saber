from __future__ import print_function
import os
import argparse
import configparser

def config_parser(config_filepath):
    """ Returns a parsed config file object

    Args:
        config_filepath: filepath to .ini config file
    """
    config = configparser.ConfigParser()
    config.read(config_filepath)
    return config

def process_parameters(config, cli_arguments):
    """ Load parameters from ini file if specificed.

    Loads parameters from the ini file if specified, take into account any
    command line argument, and ensure that each parameter is cast to the correct
    type. Command line arguments take precedence over parameters specified in
    the parameter file.

    Args:
        parameters_filepath: path to ini file containing the parameters
    """
    parameters = {}

    parameters['debug'] = bool('True' == config['settings']['debug'])
    parameters['train_model'] = bool(config['settings']['train_model'])

    parameters['dataset_text_folder'] = str(config['dataset']['dataset_text_folder'])
    parameters['output_folder'] = str(config['dataset']['output_folder'])

    parameters['activation_function'] = str(config['hyperparameters']['activation_function'])
    parameters['batch_size'] = int(config['hyperparameters']['batch_size'])
    parameters['dropout_rate'] = float(config['hyperparameters']['dropout_rate'])
    parameters['gradient_clipping_value'] = float(config['hyperparameters']['gradient_clipping_value'])
    parameters['learning_rate'] = float(config['hyperparameters']['learning_rate'])
    parameters['maximum_number_of_epochs'] = int(config['hyperparameters']['maximum_number_of_epochs'])
    parameters['optimizer'] = str(config['hyperparameters']['optimizer'])
    parameters['max_seq_len'] = int(config['hyperparameters']['max_seq_len'])

    for key, value in cli_arguments.items():
        if value is not None:
            parameters[key] = value

    return parameters

def parse_arguments():
    """ Parse command line arguments passed with call to Kari.

    Returns:
        cli_arguments: a dictionary of parsed CL arguments.
    """

    parser = argparse.ArgumentParser(description='Kari CLI')

    parser.add_argument('--config_filepath',
                        default=os.path.join('.', 'config.ini'),
                        help = '''path to the .ini file containing the
                        parameters. Defaults to './config.ini''')

    parser.add_argument('--activation_function', required=False, type=str, help='')
    parser.add_argument('--batch_size', required=False, type=int, help='')
    # parser.add_argument('--character_embedding_dimension', default=argument_default_value, help='')
    # parser.add_argument('--character_lstm_hidden_state_dimension', default=argument_default_value, help='')
    # parser.add_argument('--check_for_digits_replaced_with_zeros', default=argument_default_value, help='')
    # parser.add_argument('--check_for_lowercase', default=argument_default_value, help='')
    parser.add_argument('--dataset_text_folder', required=False, type=str, help='')
    parser.add_argument('--debug', required=False, type=bool, help='')
    parser.add_argument('--dropout_rate', required=False, type=float, help='')
    # parser.add_argument('--freeze_token_embeddings',   default=argument_default_value, help='')
    parser.add_argument('--gradient_clipping_value', required=False, type=float, help='')
    parser.add_argument('--learning_rate', required=False, type=float, help='')
    # parser.add_argument('--load_only_pretrained_token_embeddings',   default=argument_default_value, help='')
    # parser.add_argument('--load_all_pretrained_token_embeddings',   default=argument_default_value, help='')
    # parser.add_argument('--main_evaluation_mode',   default=argument_default_value, help='')
    parser.add_argument('--maximum_number_of_epochs', required=False, type=int, help='')
    # parser.add_argument('--number_of_cpu_threads',   default=argument_default_value, help='')
    # parser.add_argument('--number_of_gpus',   default=argument_default_value, help='')
    parser.add_argument('--optimizer', required=False, type=str, help='')
    parser.add_argument('--output_folder', required=False, type=str, help='')
    # parser.add_argument('--patience', default=argument_default_value, help='')
    # parser.add_argument('--plot_format', default=argument_default_value, help='')
    # parser.add_argument('--pretrained_model_folder', default=argument_default_value, help='')
    # parser.add_argument('--reload_character_embeddings', default=argument_default_value, help='')
    # parser.add_argument('--reload_character_lstm', default=argument_default_value, help='')
    # parser.add_argument('--reload_crf', default=argument_default_value, help='')
    # parser.add_argument('--reload_feedforward', default=argument_default_value, help='')
    # parser.add_argument('--reload_token_embeddings', default=argument_default_value, help='')
    # parser.add_argument('--reload_token_lstm', default=argument_default_value, help='')
    # parser.add_argument('--remap_unknown_tokens_to_unk', default=argument_default_value, help='')
    # parser.add_argument('--spacylanguage', default=argument_default_value, help='')
    # parser.add_argument('--tagging_format', default=argument_default_value, help='')
    # parser.add_argument('--token_embedding_dimension', default=argument_default_value, help='')
    # parser.add_argument('--token_lstm_hidden_state_dimension', default=argument_default_value, help='')
    # parser.add_argument('--token_pretrained_embedding_filepath', default=argument_default_value, help='')
    # parser.add_argument('--tokenizer', default=argument_default_value, help='')
    parser.add_argument('--train_model', required=False, type=bool, help='')
    parser.add_argument('--max_seq_len', required=False, type=int, help='')
    # parser.add_argument('--use_character_lstm', default=argument_default_value, help='')
    # parser.add_argument('--use_crf', default=argument_default_value, help='')
    # parser.add_argument('--use_pretrained_model', default=argument_default_value, help='')
    # parser.add_argument('--verbose', default=argument_default_value, help='')

    try:
        cli_arguments = parser.parse_args()
    except:
        parser.print_help()
        sys.ext(0)

    cli_arguments = vars(cli_arguments)
    return cli_arguments
