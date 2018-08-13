#!/usr/bin/env python3
import logging
import os
import sys
import traceback

from saber.config import Config
from saber.sequence_processor import SequenceProcessor

def main():
    """Coordinates a complete training cycle, including reading in a config, loading dataset(s),
    training the model, and saving the models weights."""
    # create and collect model and training parameters
    config = Config(cli=True)

    # currently performs training by default
    sp = SequenceProcessor(config)
    sp.load_dataset()

    # if pretrained token embeddings are provided, load them
    if config.pretrained_embeddings:
        sp.load_embeddings()
    sp.create_model()
    sp.fit()

    # save the model
    if config.save_model:
        ds_names = [os.path.basename(x) for x in config.dataset_folder]
        saved_model_dir = os.path.join(config.output_folder, '_'.join(ds_names))
        sp.save(saved_model_dir)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nQutting Saber...")
        logging.warn('Saber was terminated early due to KeyboardInterrupt')
