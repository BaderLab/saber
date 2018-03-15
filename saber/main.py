from utils_parameter_parsing import *
from sequence_processor import SequenceProcessor

# TODO (johngiorgi): do something about paths as arguments - normalize?

def main():
    """ Saber main method. """
    cli_arguments = parse_arguments() # parse CL args
    config = config_parser(cli_arguments['config_filepath']) # parse config.ini
    # resolve parameters, cast to correct types
    parameters = process_parameters(config, cli_arguments)

    # currently performs training by default
    sequence_processor = SequenceProcessor(parameters)
    sequence_processor.load_dataset()
    sequence_processor.create_model()
    sequence_processor.fit()

if __name__ == '__main__':
    main()
