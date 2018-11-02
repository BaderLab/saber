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
    config = Config(cli=True)
    saber = Saber(config)

    if config.pretrained_model:
        saber.load(config.pretrained_model)

    saber.load_dataset()

    # don't build a new model if pre-trained one was provided
    if not config.pretrained_model:
        # don't load embeddings if a pre-trained model was provided
        if config.pretrained_embeddings:
            saber.load_embeddings()
        saber.build()

    try:
        saber.train()
    except KeyboardInterrupt:
        print("\nQutting Saber...")
        logging.warning('Saber was terminated early due to KeyboardInterrupt')
    finally:
        if config.save_model:
            saber.save()

if __name__ == '__main__':
    main()
