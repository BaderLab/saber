#!/usr/bin/env python3
"""Simple web service which exposes Saber's functionality via a RESTful API.
"""
import argparse

from flask import Flask, jsonify, redirect, request
from waitress import serve

from .. import constants
from ..utils import app_utils

app = Flask(__name__)

saber = app_utils.load_models(constants.ENTITIES)

print('Saber version: {0}'.format(constants.__version__))


@app.route('/')
def serve_api_docs():
    """Flask view that redirects to the Saber API docs from route '/'.
    """
    return redirect('https://baderlab.github.io/saber-api-docs/')


@app.route('/annotate/text', methods=['POST'])
def annotate_text():
    """Annotates raw text recieved in a POST request.

    Returns:
        JSON formatted string.
    """
    parsed_request_json = app_utils.parse_request_json(request)

    # raw text to perform annotation on
    text = parsed_request_json['text']

    annotation = predict(text=text,
                         coref=parsed_request_json['coref'],
                         ground=parsed_request_json['ground'])

    return jsonify(annotation)


@app.route('/annotate/pmid', methods=['POST'])
def annotate_pmid():
    """Annotates the abstract of a document with the PubMed ID recieved in a POST request.

    Returns:
        JSON formatted string.
    """
    parsed_request_json = app_utils.parse_request_json(request)

    # use Entrez Utilities Web Service API to get the abtract text
    title, abstract = app_utils.get_pubmed_text(parsed_request_json['pmid'])
    text = '{}\n{}'.format(title, abstract)

    annotation = predict(text=text,
                         coref=parsed_request_json['coref'],
                         ground=parsed_request_json['ground'])

    return jsonify(annotation)


def predict(text, coref=False, ground=False):
    """Annotates raw text (`text`) for entities according to their boolean value in `ents`.

    Args:
        text (str): Raw text to be annotated.
        ents: Dictionary of entity, boolean pairs representing whether or not to annotate the text
            for the given entities.

    Returns:
        Dictionary containing the annotated entities and processed text.
    """
    ann = saber.annotate(text, coref=coref, ground=ground)

    return ann


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Saber web-service.')
    parser.add_argument('-p', '--port', help='Port number. Defaults to 5000', type=int, default=5000)
    args = vars(parser.parse_args())

    serve(app, host='0.0.0.0', port=args['port'])
