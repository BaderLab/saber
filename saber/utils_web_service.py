import binascii

import xml.etree.ElementTree as ET
from urllib.request import urlopen

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

def is_gz_file(filepath):
    """Returns true if the file at filepath is gzip compressed."""
    with open(filepath, 'rb') as test_f:
        return binascii.hexlify(test_f.read(2)) == b'1f8b'
