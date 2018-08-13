#!/usr/bin/env python3
"""A simple script to train a model with Saber.

Run the script with:
```
python -m saber.train
```
e.g.
```
python -m saber.train --dataset_folder ./datasets/NCBI_disease_BIO --epochs 25
```
"""
import logging

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

    try:
        sp.fit()
    except KeyboardInterrupt:
        print("\nQutting Saber...")
        logging.warning('Saber was terminated early due to KeyboardInterrupt')
    finally:
        # save the model
        if config.save_model:
            sp.save()

if __name__ == '__main__':
    main()
