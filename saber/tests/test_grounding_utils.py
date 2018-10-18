"""Any and all unit tests for the grounding_utils (saber/utils/grounding_utils.py).
"""
import pytest

from ..utils import grounding_utils


@pytest.fixture
def annotation():
    """Returns a dictionary object similar to one that might be returned by `Saber.annotate()`
    """
    annotation = {"ents": [{"text": "Hdm2", "label": "PRGE", "start": 23, "end": 27},
                           {"text": "MK2", "label": "PRGE", "start": 31, "end": 34},
                           {"text": "p53", "label": "PRGE", "start": 66, "end": 69}
                          ],
                  "text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.",
                  "title": ""}

    return annotation

def test_query_uniprot(annotation):
    """
    """
    text = annotation['ents'][1]['text']
    actual = grounding_utils._query_uniprot(text, 9606, limit=1)
    assert len(actual) == 1
    assert actual[0]['Entry'] == 'P49137'
    assert len(actual[0].keys()) == 3
    #short text
    actual = grounding_utils._query_uniprot("p")
    assert actual == []

def test_ground(annotation):
    """
    """
    actual = grounding_utils.ground(annotation, ('human', 'mouse'), limit=2)
    assert actual is not None
    for ent in actual['ents']:
        assert 'xrefs' in ent.keys()

def test_query_hgnc(annotation):
    """
    """
    actual = grounding_utils._query_hgnc(annotation)
    assert actual is None
    #TODO: implement _query_hgnc, make test fail, then fix
