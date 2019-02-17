"""A collection of helper/utility functions for grounding entities.
"""
import logging

import requests


LOGGER = logging.getLogger(__name__)

def ground(annotation):
    """Maps entities in `annotation` to unique indentifiers in an external database or ontology.

    For each entry in `annotation[ents]`, the text representing the annotation (`ent['text']`) is
    mapped to a unique identifier in an external database or ontology (if such a unique identifier
    is found). Each annotation in `annotation` is updated with an 'xrefs' key which contains a
    dictionary with information representing the mapping.

    This function relies on the EXTRACT API to perform the mapping.

    Args:
        annotation (dict): A dict containing a list of annotations at key 'ents'. Each annotation
            is expected to have a key 'text'.

    Resources:
        - EXTRACT 2.0 API: https://extract.jensenlab.org/
    """
    request = 'https://tagger.jensenlab.org/GetEntities?format=tsv&document='

    for ent in annotation['ents']:

        current_request = '{}{}'.format(request, ent['text'])

        # need to specify entity types for anything but PRGE
        if ent['label'] == 'CHED':
            current_request += '&entity_types=-1'
        elif ent['label'] == 'DISO':
            current_request += '&entity_types=-26'
        elif ent['label'] == 'LIVB':
            current_request += '&entity_types=-2'

        r = requests.get(current_request)
        response = r.text

        if response:
            xrefs = []
            entries = response.split('\n')

            for entry in entries:
                _, organism_id, entry_id = entry.split('\t')

                xref = {'namespace': 'TODO', 'id': entry_id}

                if int(organism_id) > 0:
                    xref['organism-id'] = organism_id

                xrefs.append(xref)

            ent.update(xrefs=xrefs)

    return annotation
