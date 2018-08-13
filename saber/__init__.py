from datetime import datetime
import os
import logging

from . import constants

print('Saber version: {0}'.format(constants.__version__))

# Set Tensforflow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# If applicable, delete the existing log file to generate a fresh log file during each execution
try:
    os.remove("saber.log")
except OSError:
    pass

# Create the Logger
logging.basicConfig(filename="saber.log",
                    level=logging.DEBUG,
                    format='%(name)s - %(levelname)s - %(message)s')

# Log the date to start
logging.info('Saber invocation: %s\n%s', datetime.now(), '=' * 75)
