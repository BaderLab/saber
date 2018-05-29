# -*- coding: utf-8 -*-
import json
import utils_app

from flask import request
from flask import jsonify
from flask import Flask
app = Flask(__name__)

ENTITIES = {'ANAT': False,
            'CHED': True,
            'DISO': False,
            'LIVB': True,
            'PRGE': True,
            'TRIG': False
            }

MODELS = utils_app.load_models(ENTITIES)

@app.route('/annotate/text', methods=['POST'])
def annotate_text():
    """Annotates raw text recieved in a POST request.

    Returns:
        a json formatted string
    """
    # force=True means Content-Type does not necc. have to be application/json
    # so long as json in POST request is valid.
    data = request.get_json(force=True)
    # get args from request json
    text = data.get('text', '')
    requested_ents = data.get('ents', None)

    # decide which entities to annotate
    ents = ENTITIES
    if requested_ents is not None:
        ents = utils_app.harmonize_entities(ents, requested_ents)

    annotation = predict(text, ents)

    return jsonify(annotation)

@app.route('/annotate/pmid', methods=['POST'])
def annotate_pmid():
    """Annotates the abstract of the document with the given PubMed ID
    recieved in a POST request.

    Returns:
        a json formatted string
    """
    data = request.get_json(force=True)
    # get args from request json
    pmid = data['pmid']
    requested_ents = data.get('ents', None)

    # decide which entities to annotate
    ents = ENTITIES
    if requested_ents is not None:
        ents = utils_app.harmonize_entities(ents, requested_ents)

    # use Entrez Utilities Web Service API to get the abtract text
    _, abstract = utils_app.get_pubmed_text(pmid)

    annotation = predict(abstract, ents)

    return jsonify(annotation)

def predict(text, ents):
    """Annotates raw text for entities according to their boolean value in ents

    Args:
        text (str): raw text to be annotated
        ents: dictionary of entity, boolean key, value pairs representing
            whether or not to annotate the text for the given entities

    Returns:
        dict containing the annotated entities and processed text.
    """
    annotations = []

    for k, v in ents.items():
        if v:
            annotations.append(MODELS[k].predict(text))

    if len(annotations) == 1:
        final_annotation = annotations[0]
    elif len(annotations) > 1:
        # load json strings as dicts and create combined entity list
        combined_ents = []
        for ann in annotations:
            combined_ents.extend(ann['ents'])
        # create json containing combined annotation
        final_annotation = annotations[0]
        final_annotation['ents'] = combined_ents

    return final_annotation

if __name__ == '__main__':
    app.run(host='0.0.0.0')
