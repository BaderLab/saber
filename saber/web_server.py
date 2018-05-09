from flask import Flask
from flask import request
app = Flask('saber')

from utils_parameter_parsing import *
from sequence_processor import SequenceProcessor

PRETRAINED_MODEL = '../pretrained_models/CRAFT'
PATH_TO_CONFIG = './config.ini' # set the path to your config here

config = config_parser(PATH_TO_CONFIG) # parse config
parameters = process_parameters(config) # get parameters

sp = SequenceProcessor(parameters)
sp.load(PRETRAINED_MODEL)

@app.route('/predict/')
def predict():
    """Web service"""
    raw_text = request.args.get('text', '')
    annotation = sp.predict(raw_text)

    return annotation
