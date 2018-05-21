import json

from flask import Flask
from flask import request
from flask import jsonify
app = Flask('saber')

import utils_web_service
from config import Config
from sequence_processor import SequenceProcessor

ENTITIES = {'PRGE': True,
            'LIVB': False,
            'CHED': False,
            'DISO': False,
            'TRIG': False
            }
MODELS = utils_web_service.load_models(ENTITIES)

@app.route('/annotate/text', methods=['POST'])
def annotate_text():
    """"""
    # force=True means Content-Type does not necc. have to be application/json
    # so long as json in POST request is valid.
    data = request.get_json(force=True)
    # get args from request json
    text = data.get('text', '')
    requested_ents = data.get('ents', None)

    # decide which entities to annotate
    ents = ENTITIES
    if requested_ents is not None:
        ents = utils_web_service.harmonize_entities(ents, requested_ents)

    annotation = predict(text, ents)

    return annotation


@app.route('/annotate/pmid', methods=['POST'])
def annotate_pmid():
    """"""
    data = request.get_json(force=True)
    # get args from request json
    pmid = data['pmid']
    requested_ents = data.get('ents', None)

    # decide which entities to annotate
    ents = ENTITIES
    if requested_ents is not None:
        ents = utils_web_service.harmonize_entities(ents, requested_ents)

    # use Entrez Utilities Web Service API to get the abtract text
    _, abstract = utils_web_service.get_pubmed_text(pmid)

    annotation = predict(abstract, ents)

    return annotation

def predict(text, ents):
    """"""
    annotations = []

    for k, v in ents.items():
        if v:
            annotations.append(MODELS[k].predict(text))

    if len(annotations) == 1:
        final_annotation = annotations[0]
    elif len(annotations) > 1:
        # load json strings as dicts and create combined entity list
        combined_ents = sum([json.loads(ann)['ents'] for ann in annotations])
        # create json containing combined annotation
        final_annotation = annotations[0]
        final_annotation['ents'] = combined_ents
        final_annotation = jsonify(final_annotation)

    return final_annotation
