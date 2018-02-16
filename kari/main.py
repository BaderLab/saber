# import pandas as pd
# import numpy as np
from utils_parameter_parsing import *
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

def main():
    """ Kari main method. """
    cli_arguments = parse_arguments() # parse CL args
    config = config_parser(cli_arguments['config_filepath']) # parse config.ini
    # resolve parameters, cast to correct types
    parameters = process_parameters(config, cli_arguments)

    # https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/
    sequence_processing_model = SequenceProcessingModel(**parameters)
    sequence_processing_model.fit()
    sequence_processing_model.predict()

if __name__ == '__main__':
    main()
