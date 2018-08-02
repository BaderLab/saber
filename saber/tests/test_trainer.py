# TODO (johngiorgi): add a dummy model fixture
# TODO (johngiorgi): begin writing tests, start with _split_train_valid

import os
import pytest

from ..config import Config
from ..dataset import Dataset
from ..trainer import Trainer

# constants
PATH_TO_DUMMY_CONFIG = os.path.abspath('saber/tests/resources/dummy_config.ini')
PATH_TO_DUMMY_DATASET = os.path.abspath('saber/tests/resources/dummy_dataset_1')

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object after parsing the dummy config file."""
    # parse the dummy config
    return Config(filepath=PATH_TO_DUMMY_CONFIG, cli=False)

@pytest.fixture
def dummy_dataset():
    """Returns a single dummy Dataset instance after calling load_dataset()"""
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(PATH_TO_DUMMY_DATASET, replace_rare_tokens=False)
    dataset.load_dataset()

    return dataset

@pytest.fixture
def dummy_trainer(dummy_config, dummy_dataset):
    dummy_trainer = Trainer(config=dummy_config, ds=dummy_dataset, model=None)
    return dummy_trainer

def test_split_train_valid(dummy_trainer):
    """"""
    empty_test = {}
    missing_keys_test = {"Missing": 1, "the": 2, "right": 5, "keys!": 6}
    with pytest.raises(ValueError):
        dummy_trainer._split_train_valid(empty_test)
    with pytest.raises(ValueError):
        dummy_trainer._split_train_valid(missing_keys_test)
