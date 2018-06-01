import os
import sys
import argparse
import configparser

from preprocessor import Preprocessor

class Config(object):
    def __init__(self, config_filepath='./config.ini', cli=False):
        # a parsed python configparser object
        self.config = None

        # parse cli arguments (if they exist)
        self.cli_arguments = {}
        if cli:
            self.parse_cli_args()
        self.parse_config_args(config_filepath)
        # harmonize cli and config arguments and apply post processing
        self.process_parameters(self.cli_arguments)

    def parse_config_args(self, config_filepath):
        """Returns a parsed config file object.

        Args:
            config_filepath: filepath to .ini config file
        """
        config = configparser.ConfigParser()
        config.read(config_filepath)

        self.config = config

        return True

    def process_parameters(self, cli_arguments):
        """Load parameters from ini file if specificed.

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
        parameters['model_name'] = self.config['mode']['model_name']
        parameters['train_model'] = self.config['mode'].getboolean \
            ('train_model')
        parameters['load_pretrained_model'] = self.config['mode'].getboolean \
            ('load_pretrained_model')

        # data
        parameters['dataset_folder'] = self.config['data']['dataset_folder']. \
            split(',')
        parameters['output_folder'] = self.config['data']['output_folder']
        parameters['pretrained_model_weights'] = self.config['data'] \
            ['pretrained_model_weights']
        parameters['token_pretrained_embedding_filepath'] = self.config \
            ['data']['token_pretrained_embedding_filepath']

        # model
        parameters['token_embedding_dimension'] = self.config['model'].getint( \
            'token_embedding_dimension')
        parameters['character_embedding_dimension'] = self.config['model']. \
            getint('character_embedding_dimension')

        # training
        parameters['optimizer'] = self.config['training']['optimizer']
        parameters['activation_function'] = self.config['training'] \
            ['activation_function']
        parameters['learning_rate'] = self.config['training'].getfloat \
            ('learning_rate')
        parameters['decay'] = self.config['training'].getfloat('decay')
        parameters['gradient_normalization'] = self.config['training']. \
            getfloat('gradient_normalization')
        parameters['dropout_rate'] = self.config['training']['dropout_rate']. \
            split(',')
        parameters['batch_size'] = self.config['training'].getint('batch_size')
        parameters['k_folds'] = self.config['training'].getint('k_folds')
        parameters['maximum_number_of_epochs'] = self.config['training']. \
            getint('maximum_number_of_epochs')

        # advanced
        parameters['verbose'] = self.config['advanced'].getboolean('verbose')
        parameters['debug'] = self.config['advanced'].getboolean('debug')
        parameters['replace_rare_tokens'] = self.config['advanced'].getboolean \
            ('replace_rare_tokens')
        parameters['trainable_token_embeddings'] = self.config['advanced']. \
            getboolean('trainable_token_embeddings')

        # overwrite any parameters in the config if specfied at CL
        for key, value in cli_arguments.items():
            if value is not None:
                parameters[key] = value

        # do any post-processing here
        # replace all whitespace with single space, create list of filepaths
        parameters['dataset_folder'] = [Preprocessor.sterilize(ds) for ds in
            parameters['dataset_folder']]

        # convert dropout rates to floats
        parameters['dropout_rate'] = {
            'input': float(parameters['dropout_rate'][0]),
            'output': float(parameters['dropout_rate'][1]),
            'recurrent': float(parameters['dropout_rate'][2]),
            'word_embed':float(parameters['dropout_rate'][3])
        }

        # normalize all str arguments (expect directory/file paths)
        parameters['model_name'] = Preprocessor.sterilize(parameters \
            ['model_name'])
        parameters['optimizer'] = Preprocessor.sterilize(parameters \
            ['optimizer']).lower()
        parameters['activation_function'] = Preprocessor.sterilize(parameters \
            ['activation_function']).lower()

        # do not use gradient normalization if config value is 0
        if parameters['gradient_normalization'] == 0:
            parameters['gradient_normalization'] = None
        if parameters['token_pretrained_embedding_filepath'] == '':
            parameters['token_pretrained_embedding_filepath'] = None

        # use parameters dictionary to update instance attributes
        for k, v in parameters.items():
            setattr(self, k, v)

        return True

    def parse_cli_args(self):
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
        parser.add_argument('--debug', required=False, action='store_true', help='')
        parser.add_argument('--decay', required=False, type=float, help='')
        parser.add_argument('--dropout_rate', required=False, type=dict, help='')
        parser.add_argument('--trainable_token_embeddings', required=False, action='store_true', help='')
        parser.add_argument('--gradient_normalization', required=False, type=float, help='')
        parser.add_argument('--k_folds', required=False, type=int, help='')
        parser.add_argument('--learning_rate', required=False, type=float, help='')
        parser.add_argument('--load_pretrained_model', required=False, action='store_true', help='')
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
        parser.add_argument('--replace_rare_tokens', required=False, action='store_true', help='')
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
        parser.add_argument('--train_model', required=False, action='store_true', help='')
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
        self.cli_arguments = cli_arguments

        return True
