"""A collection of helper/utility functions for grounding entities.
"""
import logging

import requests

from .. import constants

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

    # collect annotations made by Saber in a dictionary
    annotations = {
        'CHED': [ent for ent in annotation['ents'] if ent['label'] == 'CHED'],
        'DISO': [ent for ent in annotation['ents'] if ent['label'] == 'DISO'],
        'LIVB': [ent for ent in annotation['ents'] if ent['label'] == 'LIVB'],
        'PRGE': [ent for ent in annotation['ents'] if ent['label'] == 'PRGE'],
    }

    for label, anns in annotations.items():
        if anns:
            # prepand to GET request the text to ground along with its entity type
            current_request = '{}{}'.format(request, '+'.join([ann['text'] for ann in anns]))
            if label in constants.ENTITY_TYPES:
                current_request += '&entity_types={}'.format(constants.ENTITY_TYPES[label])

            # request to EXTRACT 2.0 API
            response = requests.get(current_request).text
            entries = [entry.split('\t') for entry in response.split('\n')] if response else []

            xrefs = {}

            # collect unique identifiers returned by EXTRACT 2.0 API
            for entry in entries:
                xref = {'namespace': constants.NAMESPACES[label], 'id': entry[-1]}
                # in the future, EXTRACT 2.0 API will to assign organism-ids to PRGE labels
                if label == 'PRGE':
                    xref['organism-id'] = entry[1]

                if entry[0] in xrefs:
                    xrefs[entry[0]].append(xref)
                else:
                    xrefs[entry[0]] = [xref]

            # update annotations with xrefs field
            for ann in anns:
                if ann['text'] in xrefs:
                    ann.update(xrefs=xrefs[ann['text']])

    return annotation
