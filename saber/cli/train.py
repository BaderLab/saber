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
from ..constants import __version__
from ..saber import Saber

print('Saber version: {0}'.format(__version__))


def main():
    """Coordinates a complete training cycle, including reading in a config, loading dataset(s),
    training the model, and saving the models weights."""
    config = Config(cli=True)
    saber = Saber(config)

    if config.pretrained_model:
        saber.load(config.pretrained_model)

    saber.load_dataset()

    # Don't build a new model if pre-trained one was provided
    if not config.pretrained_model:
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
