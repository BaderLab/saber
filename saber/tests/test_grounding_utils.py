import pytest
# import json

from ..utils import grounding_utils

@pytest.fixture
def annotation():
    return {
        "ents": [
            {
                "end": 27,
                "label": "PRGE",
                "start": 23,
                "text": "Hdm2"
            },
            {
                "end": 34,
                "label": "PRGE",
                "start": 31,
                "text": "MK2"
            },
            {
                "end": 69,
                "label": "PRGE",
                "start": 66,
                "text": "p53"
            }
        ],
        "text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.",
        "title": ""
    }


def test_query_uniprot(annotation):
    text = annotation['ents'][1]['text']
    r = grounding_utils._query_uniprot(text,9606,limit=1)
    assert len(r) == 1
    assert r[0]['Entry'] == 'P49137'
    assert len(r[0].keys()) == 3
    #short text
    r = grounding_utils._query_uniprot("p")
    assert r == []


def test_ground(annotation):
    a = grounding_utils.ground(annotation, ('human','mouse'), limit=2)
    assert a != None
    for ent in a['ents']:
        assert 'xrefs' in ent.keys()


def test_query_hgnc(annotation):
    a = grounding_utils._query_hgnc(annotation)
    assert a == None
    #TODO: implement _query_hgnc, make test fail, then fix
