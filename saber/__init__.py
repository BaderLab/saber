import logging
import os
from datetime import datetime
# make Saber, the interface for the entire package, importable from root
from .saber import Saber

# if applicable, delete the existing log file to generate a fresh log file during each execution
try:
    os.remove("saber.log")
except OSError:
    pass

# create the logger
logging.basicConfig(filename="saber.log",
                    level=logging.DEBUG,
                    format='%(name)s - %(levelname)s - %(message)s')

# log the date to start
logging.info('Saber invocation: %s\n%s', datetime.now(), '=' * 75)
