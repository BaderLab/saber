"""Any and all unit tests for the grounding_utils (saber/utils/grounding_utils.py).
"""
import pytest

from ..utils import grounding_utils

@pytest.fixture
def ched_annotation():
    """
    """
    annotation = {"ents": [{"text": "glucose", "label": "CHED", "start": 0, "end": 6}],
                  "text": "glucose",
                  "title": ""}
    return annotation

@pytest.fixture
def diso_annotation():
    """
    """
    annotation = {"ents": [{"text": "cancer", "label": "DISO", "start": 0, "end": 5}],
                  "text": "cancer",
                  "title": ""}
    return annotation

@pytest.fixture
def livb_annotation():
    """
    """
    annotation = {"ents": [{"text": "mouse", "label": "LIVB", "start": 0, "end": 4}],
                  "text": "mouse",
                  "title": ""}
    return annotation

@pytest.fixture
def prge_annotation():
    """
    """
    annotation = {"ents": [{"text": "p53", "label": "PRGE", "start": 0, "end": 3}],
                  "text": "p53",
                  "title": ""}
    return annotation

def test_ground_chemicals(ched_annotation):
    """Asserts that `grounding_utils.ground()` returns the expected value for a simple example with
    chemical entities.
    """
    xrefs = [
        {'namespace': 'TODO', 'id': 'CIDs00005793'},
        {'namespace': 'TODO', 'id': 'CIDs10954115'},
        {'namespace': 'TODO', 'id': 'CIDs53782692'},
    ]

    actual = grounding_utils.ground(ched_annotation)
    ched_annotation['ents'][0].update(xrefs=xrefs)
    expected = ched_annotation

    assert actual == expected

def test_ground_diso(diso_annotation):
    """Asserts that `grounding_utils.ground()` returns the expected value for a simple example with
    disease entities.
    """
    xrefs = [
        {'namespace': 'TODO', 'id': 'DOID:162'},
    ]

    actual = grounding_utils.ground(diso_annotation)
    diso_annotation['ents'][0].update(xrefs=xrefs)
    expected = diso_annotation

    assert actual == expected

def test_ground_livb(livb_annotation):
    """Asserts that `grounding_utils.ground()` returns the expected value for a simple example with
    species entities.
    """
    xrefs = [
        {'namespace': 'TODO', 'id': '10090'},
        {'namespace': 'TODO', 'id': '10088'},
    ]

    actual = grounding_utils.ground(livb_annotation)
    livb_annotation['ents'][0].update(xrefs=xrefs)
    expected = livb_annotation

    assert actual == expected

def test_ground_prge(prge_annotation):
    """Asserts that `grounding_utils.ground()` returns the expected value for a simple example with
    species entities.
    """
    xrefs = [
        {'namespace': 'TODO', 'id': 'ENSP00000269305', 'organism-id': '9606'},
        {'namespace': 'TODO', 'id': '10088'},
    ]

    actual = grounding_utils.ground(prge_annotation)
    prge_annotation['ents'][0].update(xrefs=xrefs)
    expected = prge_annotation

    assert actual == expected
