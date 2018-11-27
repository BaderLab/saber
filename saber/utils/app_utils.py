"""A collection of web-service-related helper/utility functions.
"""
import logging
import os.path
import traceback
import xml.etree.ElementTree as ET
from urllib.error import HTTPError
from urllib.request import urlopen

import tensorflow as tf

from .. import constants
from ..saber import Saber
from ..utils import generic_utils

# TODO: Need better error handeling here
LOGGER = logging.getLogger(__name__)

def get_pubmed_xml(pmid):
    """Uses the Entrez Utilities Web Service API to fetch XML representation of PubMed document.

    Args:
        pmid (int): the PubMed ID of the abstract to fetch

    Returns:
        response from Entrez Utilities Web Service API

    Raises:
        ValueError if 'pmid' is not an integer
        ValueError if 'pmid' has value less than 1
        AssertionError if the requested PubMed ID, 'pmid' and the return PubMedID do not match.
    """
    if not isinstance(pmid, int):
        err_msg = "Argument 'pmid' must be of type {}, not {}.".format(int, type(pmid))
        LOGGER.error('ValueError %s', err_msg)
        raise ValueError(err_msg)
    if pmid < 1:
        err_msg = "Argument 'pmid' must have a value of 1 or greater. Got {}".format(pmid)
        LOGGER.error('ValueError %s', err_msg)
        raise ValueError(err_msg)

    try:
        request = '{}{}'.format(constants.EUTILS_API_ENDPOINT, pmid)
        response = urlopen(request).read()
    except HTTPError:
        err_msg = ("HTTP Error 400: Bad Request was returned. Check that the supplied value for "
                   "'pmid' ({}) is a valid PubMed ID.".format(pmid))
        traceback.print_exc()
        LOGGER.error('HTTPError %s', err_msg)
        print(err_msg)
    else:
        root = get_root(response)
        response_pmid = root.find('PubmedArticle').find('MedlineCitation').find('PMID').text
        # ensure that requested and returned pubmed ids are the same
        if not int(response_pmid) == pmid:
            err_msg = ('Requested PubMed ID and PubMed ID returned by Entrez Utilities Web Service '
                       'API do not match.')
            LOGGER.error('AssertionError %s', err_msg)
            raise AssertionError(err_msg)

    return response

def get_pubmed_text(pmid):
    """Returns the abstract title and text for a given PubMed id using the the Entrez Utilities Web
    Service API.

    Args:
        pmid (int): the PubMed ID of the abstract to fetch

    Returns:
        two-tuple containing the abstract title and text for PubMed ID 'pmid'
    """
    xml = get_pubmed_xml(pmid)
    root = get_root(xml)
    # TODO: There has got to be a better way to do this.
    # recurse down the xml tree to abstractText
    abstract_title = root.find('PubmedArticle').find('MedlineCitation').find('Article').find('ArticleTitle').text
    abstract_text = root.find('PubmedArticle').find('MedlineCitation').find('Article').find('Abstract').find('AbstractText').text

    return abstract_title, abstract_text

def get_root(xml):
    """Return root of given XML string.

    Args:
        xml (str): a string containing the contents of an XML file.

    Returns:
        root of the given XML file, `xml`.
    """
    return ET.fromstring(xml)

def load_models(ents):
    """Loads a model for each entity in `ents`.

    Given a dict with key (str): value (bool) pairs, loads each model (key) for which value is True.

    Args:
        ents (dict): a dictionary where the keys correspond to entities and the values are booleans.

    Returns:
        a dictionary with keys representing the model and values a loaded Saber object.
    """
    models = {} # acc for models
    for ent, value in ents.items():
        if value:
            path_to_model = os.path.join(constants.PRETRAINED_MODEL_DIR, ent)
            generic_utils.extract_directory(path_to_model)
            # create and load the pre-trained models
            saber = Saber()
            saber.load(path_to_model)
            models[ent] = saber
    # TEMP: Weird solution to a weird bug.
    # https://github.com/tensorflow/tensorflow/issues/14356#issuecomment-385962623
    # Unclear if this will work for multiple models! If not, return a graph for each.
    graph = tf.get_default_graph()

    return models, graph

def harmonize_entities(default_ents, requested_ents):
    """Harmonizes two dictionaries representing default_ents and requested requested_ents.

    Given two dictionaries of entity: boolean key: value pairs, returns a
    dictionary where the values of entities specified in `requested_ents` override those specified
    in `default_ents`. Entities present in `default_ents` but not in `requested_ents` will be set to
    False by default.

    Args:
        default_ents (dict): contains entity (str): boolean key: value pairs representing which
            entities should be annotated in a given text.
        requested_ents (dict): contains entity (str): boolean key: value pairs representing which
            entities should be predicted in a given text.

    Returns: a dictionary containing all key, value pairs in `default_ents`, where values in
        `requested_ents` override those in default_ents. Any key in `default_ents` but not in
        `requested_ents` will have its value set to False by default.
    """
    entities = {}
    for ent in default_ents:
        entities[ent] = False
    for ent, value in requested_ents.items():
        if ent in entities:
            entities[ent] = value

    return entities

def parse_request_json(request):
    """Returns a dictionary of data parsed from a JSON payload passed in a POST request to Saber.
    """
    request_json = request.get_json(force=True)
    parsed_request_json = {
        'text': request_json.get('text', None),
        'pmid': request_json.get('pmid', None),
        'ents': request_json.get('ents', None),
        'coref': request_json.get('coref', False),
        'ground': request_json.get('ground', False),
    }

    # decide which entities to annotate
    default_ents, requested_ents = constants.ENTITIES, parsed_request_json['ents']
    if requested_ents is not None:
        parsed_request_json['ents'] = harmonize_entities(default_ents, requested_ents)
    else:
        parsed_request_json['ents'] = default_ents

    return parsed_request_json

def combine_annotations(annotations):
    """Given a list of annotations made by a Saber model, combines all annotations under one dict.

    Args:
        annotations (list): a list of annotations returned by a Saber model

    Returns:
        a dict containing all annotations in `annotations`.
    """
    combined_anns = []
    for ann in annotations:
        combined_anns.extend(ann['ents'])
    # create json containing combined annotation
    return combined_anns
