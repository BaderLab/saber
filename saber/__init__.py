import logging
import os
from datetime import datetime

# Make Saber, the interface for the entire package, importable from root
from .saber import Saber
from .config import Config

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
