import logging

import requests


def _query_uniprot(text, organisms=('9606'), limit=1):
    """Query for accession numbers using UniProt REST api
    Example:
      https://www.uniprot.org/uniprot/?query=name:mk2&columns=id&format=tab
    See:
      https://www.uniprot.org/help/api_queries
      https://www.uniprot.org/help/query-fields
      https://www.uniprot.org/help/uniprotkb_column_names
      https://www.uniprot.org/help/programmatic_access

    :param text: gene or gene product name, synonym, id
    :param organisms: names or taxonomy ids; can be number, string or tuple
    :param limit: max no. hits (result rows ordered by relevance)
    :return: xrefs - list of dictionaries [{'col1':'val1', 'col2':'val2',..},..]
    """
    if len(text) < 2:
        logging.error('query text must be at least two characters long')
        return []

    xrefs = []
    # fields, columns can be parameters too if we'd generalize later
    fields = ('name', 'gene_exact', 'mnemonic')
    columns = ('id', 'organism-id', 'genes(PREFERRED)')
    params = { "sort": "score", "format": "tab" }

    query=text
    if fields!=None:
        query=" OR ".join([f + ':' + str(text) for f in fields])
    #filter by organism
    if organisms!=None:
        if isinstance(organisms, str) or isinstance(organisms, int):
            subquery="organism:"+str(organisms)
        else:
            subquery = " OR ".join(['organism:'+str(o) for o in organisms])
        query="({query}) AND ({subquery})".format(query=query,subquery=subquery)
    params.update(query=query)

    #set output data columns
    if columns!=None:
        params.update(columns=",".join(columns))

    #max no. result rows (hits)
    if limit != None:
        params.update(limit=limit)

    try:
        r = requests.get('https://www.uniprot.org/uniprot/', params=params)
        if (r.status_code == 200):
            lines = r.text.splitlines()
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
def ground(annotation, organisms=(9606), limit=10):
    copy = {k:v for k,v in annotation.items()}
    for ent in copy['ents']:
        if ent['label'] == 'PRGE':
            xrefs = _query_uniprot(ent['text'], organisms, limit)
            ent.update(xrefs=xrefs)
    return copy


# TODO try with HGNC rest api
def _query_hgnc(text):
    return None
