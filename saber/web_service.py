import os.path
import json
import zipfile

from flask import Flask
from flask import request
from flask import jsonify
app = Flask('saber')

from utils_parameter_parsing import *
from sequence_processor import SequenceProcessor
from utils_web_service import get_pubmed_text
from utils_web_service import is_gz_file

PRETRAINED_MODEL_BASE_DIR = '../pretrained_models/'
MODEL = os.path.join(PRETRAINED_MODEL_BASE_DIR, 'CRAFT')
# TRIG_MODEL = os.path.join(PRETRAINED_MODEL_BASE_DIR, 'TRIG')

if not os.path.isdir(MODEL):
    print('[INFO] Unzipping pretrained model...', end='', flush=True)
    zip_ref = zipfile.ZipFile(MODEL + '.zip', 'r')
    zip_ref.extractall(PRETRAINED_MODEL_BASE_DIR)
    zip_ref.close()
    print(' Done.')

PATH_TO_CONFIG = './config.ini' # set the path to your config here

config = config_parser(PATH_TO_CONFIG) # parse config
parameters = process_parameters(config) # get parameters

sp = SequenceProcessor(parameters)
# trig_sp = SequenceProcessor(parameters)

sp.load(MODEL)
# trig_sp.load(TRIG_MODEL)

@app.route('/annotate/text', methods=['POST'])
def annotate():
    """Web service"""
    # force=True means Content-Type does not necc. have to be application/json
    # so long as json in POST request is valid.
    data = request.get_json(force=True)
    # get text from request json
    # TODO: catch any key errors
    text = data['text']

    annotation = sp.predict(text) # preform prediction

    return annotation

@app.route('/annotate/pmid', methods=['POST'])
def pmid():
    # force=True means Content-Type does not necc. have to be application/json
    # so long as json in POST request is valid.
    data = request.get_json(force=True)
    # get PubMed ID from request json
    pmid = data['pmid']

    # use Entrez Utilities Web Service API to get the abtract text
    _, abstract = get_pubmed_text(pmid)

    # preform prediction
    annotation = json.loads(sp.predict(abstract))
    # triggers = json.loads(trig_sp.predict(abstract))

    # combine entities
    # combined_entities = genes_proteins['ents'] + triggers['ents']

    # build annotation from combined entities
    # annotation = genes_proteins
    # annotation['ents'] = combined_entities

    return jsonify(annotation)
