import os.path
import json
import zipfile
from setuptools.archive_util import unpack_archive

from flask import Flask
from flask import request
from flask import jsonify
app = Flask('saber')

from config import Config
from sequence_processor import SequenceProcessor
from utils_web_service import get_pubmed_text

PRETRAINED_MODEL_BASE_DIR = '../pretrained_models/'
MODEL = os.path.join(PRETRAINED_MODEL_BASE_DIR, 'PRGE')

# decompress the pre-trained model if this is not already done
if not os.path.isdir(MODEL):
    print('[INFO] Unzipping pretrained model...', end='', flush=True)
    unpack_archive(MODEL + '.tar.bz2', PRETRAINED_MODEL_BASE_DIR)
    print(' Done.')

config = Config() # parse config
sp = SequenceProcessor(config) # create sequence processor
sp.load(MODEL) # load the pre-trained model

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
