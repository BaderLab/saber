import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

from .config import Config
from .constants import RANDOM_STATE
# Make Saber, the interface for the entire package, importable from root
from .saber import Saber

# Fix random seeds
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# If applicable, delete the existing log file to generate a fresh log file during each execution
try:
    os.remove("saber.log")
except OSError:
    pass

# Create the logger
logging.basicConfig(filename="saber.log",
                    level=logging.DEBUG,
                    format='%(name)s - %(levelname)s - %(message)s')

# Log the date to start
logging.info('Saber invocation: %s\n%s', datetime.now(), '=' * 75)
