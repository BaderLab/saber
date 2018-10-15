#!/usr/bin/env python3
"""A simple script to train a model with Saber.

Run the script with:
```
python -m saber.cli.train
```
e.g.
```
python -m saber.cli.train --dataset_folder ./datasets/NCBI_disease_BIO --epochs 25
```
"""
import logging

from ..config import Config
from ..saber import Saber


def main():
    """Coordinates a complete training cycle, including reading in a config, loading dataset(s),
    training the model, and saving the models weights."""
    # create and collect model and training parameters
    config = Config(cli=True)

    # currently performs training by default
    saber = Saber(config)
    saber.load_dataset()

    # if pretrained token embeddings are provided, load them
    if config.pretrained_embeddings:
        saber.load_embeddings()
    saber.build()

    try:
        saber.train()
    except KeyboardInterrupt:
        print("\nQutting Saber...")
        logging.warning('Saber was terminated early due to KeyboardInterrupt')
    finally:
        # save the model
        if config.save_model:
            saber.save()

if __name__ == '__main__':
    main()
