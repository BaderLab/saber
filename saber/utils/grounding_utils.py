"""A collection of helper/utility functions for grounding entities.
"""
import logging

import requests

from ..preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)

def _query_uniprot(text, organisms=('9606'), limit=1):
    """Query for accession numbers using UniProt REST api
    Example:
      https://www.uniprot.org/uniprot/?query=name:mk2&columns=id&format=tab
    See:
      https://www.uniprot.org/help/api_queries
      https://www.uniprot.org/help/query-fields
      https://www.uniprot.org/help/uniprotkb_column_names
      https://www.uniprot.org/help/programmatic_access

     Args:
        text: Gene or gene product name, synonym, or id.
        organisms: Names or taxonomy ids; can be number, string or tuple.
        limit: Max number of hits (result rows ordered by relevance).

    Returns:
        xrefs, a list of dictionaries [{'col1':'val1', 'col2':'val2',..},..]
    """
    if len(text) < 2:
        logging.error('query text must be at least two characters long')
        return []

    xrefs = []
    # fields, columns can be parameters too if we'd generalize later
    fields = ('name', 'gene_exact', 'mnemonic')
    columns = ('id', 'organism-id', 'genes(PREFERRED)')
    params = {"sort": "score", "format": "tab"}

    query = text
    if fields is not None:
        query = " OR ".join([f + ':' + str(text) for f in fields])
    # filter by organism
    if organisms is not None:
        if isinstance(organisms, (int, str)):
            subquery = "organism:" + str(organisms)
        else:
            subquery = " OR ".join(['organism:'+str(o) for o in organisms])
        query = "({query}) AND ({subquery})".format(query=query, subquery=subquery)
    params.update(query=query)

    # set output data columns
    if columns is not None:
        params.update(columns=",".join(columns))

    # max no. result rows (hits)
    if limit != None:
        params.update(limit=limit)

    try:
        response = requests.get('https://www.uniprot.org/uniprot/', params=params)
        if response.status_code == 200:
            lines = response.text.splitlines()
            if len(lines) > 1:
                # text returned by uniprot api has weird spacing, so clean it
                col_names = [Preprocessor.sterilize(line) for line in lines[0].split('\t')]
            for line in lines[1:]:
                col_vals = line.split('\t')
                xref = dict(zip(col_names, col_vals))
                xrefs.append(xref)
        else:
            LOGGER.error('Uniprot returned: %i, params: %s', response.status_code, str(params))
    except requests.exceptions.RequestException as err:
        logging.error(err)

    return xrefs

def ground(annotation, organisms=(9606), limit=10):
    """
    """
    for ent in annotation['ents']:
        if ent['label'] == 'PRGE':
            xrefs = _query_uniprot(ent['text'], organisms, limit)
            ent.update(xrefs=xrefs)
    return annotation

# TODO try with HGNC rest api
def _query_hgnc(text):
    """
    """
    return None
