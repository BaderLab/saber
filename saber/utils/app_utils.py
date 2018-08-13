"""A collection of web-service helper/utility functions.
"""
import logging
import os.path
import traceback
import xml.etree.ElementTree as ET
from urllib.error import HTTPError
from urllib.request import urlopen

from .. import constants
from ..config import Config
from .generic_utils import decompress_model
from ..sequence_processor import SequenceProcessor

# TODO: Need better error handeling here
log = logging.getLogger(__name__)

def get_pubmed_xml(pmid):
    """Uses the Entrez Utilities Web Service API to fetch XML representation of pubmed document.

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
        log.error('ValueError %s', err_msg)
        raise ValueError(err_msg)
    if pmid < 1:
        err_msg = "Argument 'pmid' must have a value of 1 or greater. Got {}".format(pmid)
        log.error('ValueError %s', err_msg)
        raise ValueError(err_msg)

    try:
        request = '{}{}'.format(constants.EUTILS_API_ENDPOINT, pmid)
        response = urlopen(request).read()
    except HTTPError:
        err_msg = ("HTTP Error 400: Bad Request was returned. Check that the supplied value for "
                   "'pmid' ({}) is a valid PubMed ID.".format(pmid))
        traceback.print_exc()
        log.error('HTTPError %s', err_msg)
        print(err_msg)
    else:
        root = get_root(response)
        response_pmid = root.find('PubmedArticle').find('MedlineCitation').find('PMID').text
        # ensure that requested and returned pubmed ids are the same
        if not int(response_pmid) == pmid:
            err_msg = ('Requested PubMed ID and PubMed ID returned by Entrez Utilities Web Service '
                       'API do not match.')
            log.error('AssertionError %s', err_msg)
            raise AssertionError(err_msg)

    return response

def get_pubmed_text(pmid):
    """Returns the abstract title and text for a given pubmed id using the the Entrez Utilities Web
    Service API.

    Args:
        pmid (int): the PubMed ID of the abstract to fetch

    Returns:
        two-tuple containing the abstract title and text for PubMed ID 'pmid'
    """
    xml = get_pubmed_xml(pmid)
    root = get_root(xml)
    # recurse down the xml tree to abstractText
    abstract_title = root.find('PubmedArticle').find('MedlineCitation').find('Article').find('ArticleTitle').text
    abstract_text = root.find('PubmedArticle').find('MedlineCitation').find('Article').find('Abstract').find('AbstractText').text

    return abstract_title, abstract_text

def get_root(xml):
    """Return root of given XML string."""
    return ET.fromstring(xml)

def load_models(ents):
    """Loads a model for each corresponding entity in ents.

    Given a dictionary with str, bool key, value pairs, loads each model (key)
    for which value is True.

    Args:
        ents (dict): a dictionary where the keys correspond to entities and the
            values are booleans.
    Returns:
        a dictionary with keys representing the model and values a loaded
        SequenceProcessor object.

    """
    models = {} # acc for models
    config = Config() # parse config

    for k, v in ents.items():
        if v:
            path_to_model = os.path.join(constants.PRETRAINED_MODEL_BASE_DIR, k)
            # decompress the pre-trained model if this is not already done
            decompress_model(path_to_model)

            # create and load the pre-trained models
            sp = SequenceProcessor(config)
            sp.load(path_to_model)
            models[k] = sp

    return models

def harmonize_entities(default_ents, requested_ents):
    """Harmonizes two dictionaries represented default and requested entitiesself.

    Given two dictionaries of entity, boolean key value pairs, returns a
    dictionary where the values of entities specied in requested_ents override
    those specified in default_ents. Entities present in default_ents but not
    in requested_ents will be set to False by default.

    Args:
        default_ents (dict): contains entity (str), boolean key value pairs
            representing which entities should be predicted in a given text
        requested_ents (dict): contains entity (str), boolean key value pairs
            representing which entities should be predicted in a given text

    Returns: a dictionary containing all key, value pairs in default_ents,
        where values in requested_ents overide those in default_ents. Any
        key in default_ents but not in requested_ents will have its value set to
        False by default.

    """
    entities = {}
    for k in default_ents:
        entities[k] = False
    for k, v in requested_ents.items():
        if k in entities:
            entities[k] = v

    return entities
