import pytest
# import json

from ..utils import grounding

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
    r = grounding._query_uniprot(text, limit=1)
    assert len(r) == 1
    assert r[0]['Entry'] == 'P49137'
    assert len(r[0].keys()) == 3
    #short text
    r = grounding._query_uniprot("p",limit=1)
    assert r == []


def test_ground(annotation):
    a = grounding.ground(annotation)
    assert a == annotation
    #TODO: implement grounding.ground, make test fail, then fix


def test_query_hgnc(annotation):
    a = grounding._query_hgnc(annotation)
    assert a == None
    #TODO: implement _query_hgnc, make test fail, then fix
