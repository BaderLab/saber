#!/usr/bin/env python3
from config import Config
from sequence_processor import SequenceProcessor

def main():
    """Saber main method."""
    # create and collect model and training parameters
    config = Config()

    # currently performs training by default
    sp = SequenceProcessor(config)
    sp.load_dataset()

    # if pretrained token embeddings are provided, load them
    if config.token_pretrained_embedding_filepath is not None:
        sp.load_embeddings()
    sp.create_model()
    sp.fit()

if __name__ == '__main__':
    main()
