import logging

import requests


##
# Using UniProt REST api: search
# Example:
#  https://www.uniprot.org/uniprot/?query=name:mk2&columns=id&format=tab
# See:
#  https://www.uniprot.org/help/api_queries
#  https://www.uniprot.org/help/query-fields
#  https://www.uniprot.org/help/uniprotkb_column_names
#  https://www.uniprot.org/help/programmatic_access
#
# Returns a query results list [{'col1':'val1', 'col2':'val2',..},..]
##
def _query_uniprot(text, limit=10):
    xrefs = []

    # fields and columns can be args if we'd like to generalize this module
    fields = ('name', 'gene', 'mnemonic')
    columns = ('id', 'organism-id', 'genes(PREFERRED)')
    params = {
        "sort": "score",
        "format": "tab"
    }
    if len(text) < 2:
        logging.error('query text must be at least two characters long')
        return []
    elif len(fields) == 0:
        params.update(query=text)
    else:
        params.update(query=" OR ".join([f + ':' + text for f in fields]))
    if limit > 0:
        params.update(limit=limit)
    if len(columns) > 0:
        params.update(columns=",".join(columns))

    try:
        r = requests.get('https://www.uniprot.org/uniprot/', params=params)
        if (r.status_code == 200):
            lines = r.text.split('\n', limit)
            if len(lines) > 1:
                col_names = lines[0].split('\t')
            for line in lines[1:]:
                col_vals = line.split('\t')
                xref = dict(zip(col_names, col_vals))
                xrefs.append(xref)
        else:
            logging.error('Uniprot returned: {c}, params: {q}'.format(c=r.status_code, q=str(params)))
    except requests.exceptions.RequestException as e:
        logging.error(e)

    return xrefs


# TODO: find and insert 'xrefs' into the annotation doc (dict)
def ground(annotation):
    return annotation


# TODO try with HGNC rest api
def _query_hgnc(text):
    return None
