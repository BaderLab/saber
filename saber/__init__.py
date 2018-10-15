import logging
import os

from datetime import datetime

# set Tensorflow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
