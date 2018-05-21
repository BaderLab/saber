import os.path
from setuptools.archive_util import unpack_archive

import xml.etree.ElementTree as ET
from urllib.request import urlopen

from config import Config
from sequence_processor import SequenceProcessor
from constants import PRETRAINED_MODEL_BASE_DIR

"""
A collection of web-service helper/utility functions.
"""

def get_pubmed_xml(pmid):
    """Uses the Entrez Utilities Web Service API to fetch XML representation of
    pubmed document."""
    api_endpoint = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?retmode=xml&db=pubmed&id='
    response = urlopen(api_endpoint + str(pmid)).read()

    try:
        root = get_root(response)
        response_pmid = root.find('PubmedArticle').find('MedlineCitation').find('PMID').text
        # ensure that requested and returned pubmed ids are the same
        assert int(response_pmid) == pmid, 'Requested PubMed ID and PubMed ID returned by Entrez Utilities Web Service API do not match.'
        return response
    except:
        return None

def get_pubmed_text(pmid):
    """Returns the abstract title for a given pubmed id using the the Entrez
    Utilities Web Service API."""
    xml = get_pubmed_xml(pmid)
    root = get_root(xml)
    # recurse down the xml tree to abstractText
    try:
        abstract_title = root.find('PubmedArticle').find('MedlineCitation').find('Article').find('ArticleTitle').text
        abstract_text = root.find('PubmedArticle').find('MedlineCitation').find('Article').find('Abstract').find('AbstractText').text
        return (abstract_title, abstract_text)
    except:
        return None

def get_root(xml):
    """Return root of given XML string. Returns None if string could not be
    parsed."""
    try:
        return ET.fromstring(xml)
    except:
        return None

def decompress_model(filepath):
    """Decompresses a bz2 compressed Saber model.

    If filepath is not a directory, decompresses the identically named bz2 Saber
    model at filepath.
    """
    if not os.path.isdir(filepath):
        print('[INFO] Unzipping pretrained model... '.format(), end='', flush=True)
        unpack_archive(filepath + '.tar.bz2', PRETRAINED_MODEL_BASE_DIR)
        print('Done.')

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
            path_to_model = os.path.join(PRETRAINED_MODEL_BASE_DIR, k)
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
