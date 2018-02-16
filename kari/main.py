from __future__ import print_function
import os
import argparse
import configparser
# import pandas as pd
# import numpy as np

from sequence_processing_model import SequenceProcessingModel



# from dataset import Dataset
# from specify import *

# from keras_contrib.layers import CRF
# from keras.models import Model, Input
# from sklearn.model_selection import train_test_split

# TODO (johngiorgi): do something about paths as arguments - normalize?

'''
# Notes
- RNNs are tricky. Choice of batch size is important, choice of loss and
 optimizer is critical, etc. Some configurations won't converge.
-  LSTM loss decrease patterns during training can be quite different from what
you see with CNNs/MLPs/etc.
'''

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

    parameters['dropout_rate'] = float(config['hyperparameters']['dropout_rate'])
    parameters['gradient_clipping_value'] = float(config['hyperparameters']['gradient_clipping_value'])
    parameters['learning_rate'] = float(config['hyperparameters']['learning_rate'])
    parameters['maximum_number_of_epochs'] = int(config['hyperparameters']['maximum_number_of_epochs'])
    parameters['optimizer'] = str(config['hyperparameters']['optimizer'])

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


def main():
    """ Kari main method. """
    cli_arguments = parse_arguments() # parse CL args
    config = config_parser(cli_arguments['config_filepath']) # parse config.ini

    parameters = process_parameters(config, cli_arguments)

    # https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/
    # sequence_processing_model = SequenceProcessingModel(**parameters)
    # SequenceProcessingModel.fit()

if __name__ == '__main__':
    main()

    '''
    dataset = Dataset('../../datasets/NCBI_disease_train.tsv')
    dataset.load_dataset()

    # print(dataset.word_idx_sequence)

    # split the data set into train/test
    X_tr, X_te, y_tr, y_te = train_test_split(dataset.word_idx_sequence,
        dataset.tag_idx_sequence, test_size=0.1)

    model, crf = specify(dataset.word_type_count, dataset.tag_type_count,
        max_len=dataset.max_seq_len)

    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    history = model.fit(X_tr, np.array(y_tr), batch_size=10, epochs=10,
                        validation_split=0.1, verbose=1)

    hist = pd.DataFrame(history.history)
    '''
