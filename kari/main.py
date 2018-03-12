from utils_parameter_parsing import *
from sequence_processing_model import SequenceProcessor

# TODO (johngiorgi): do something about paths as arguments - normalize?
# https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/

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

    sequence_processor = SequenceProcessor(config)
    sequence_processor.load_dataset()
    sequence_processor.create_model()
    sequence_processor.fit()

if __name__ == '__main__':
    main()
