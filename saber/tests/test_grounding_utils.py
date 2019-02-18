"""Any and all unit tests for the grounding_utils (saber/utils/grounding_utils.py).
"""
import pytest

from ..utils import grounding_utils

@pytest.fixture
def blank_annotation():
    """Returns an annotation with no identified entities.
    """
    annotation = {"ents": [],
                  "text": "This is a test with no entities.",
                  "title": ""}
    return annotation

@pytest.fixture
def ched_annotation():
    """Returns an annotation with chemical entities (CHED) identified.
    """
    annotation = {"ents": [{"text": "glucose", "label": "CHED", "start": 0, "end": 0},
                           {"text": "fructose", "label": "CHED", "start": 0, "end": 0}],
                  "text": "glucose and fructose",
                  "title": ""}

    return annotation

@pytest.fixture
def diso_annotation():
    """Returns an annotation with disease entities (DISO) identified.
    """
    annotation = {"ents": [{"text": "cancer", "label": "DISO", "start": 0, "end": 0},
                           {"text": "cystic fibrosis", "label": "DISO", "start": 0, "end": 0}],
                  "text": "cancer and cystic fibrosis",
                  "title": ""}

    return annotation

@pytest.fixture
def livb_annotation():
    """Returns an annotation with species entities (LIVB) identified.
    """
    annotation = {"ents": [{"text": "mouse", "label": "LIVB", "start": 0, "end": 0},
                           {"text": "human", "label": "LIVB", "start": 0, "end": 0}],
                  "text": "mouse and human",
                  "title": ""}

    return annotation

@pytest.fixture
def prge_annotation():
    """Returns an annotation with protein/gene entities (PRGE) identified.
    """
    annotation = {"ents": [{"text": "p53", "label": "PRGE", "start": 0, "end": 0},
                           {"text": "MK2", "label": "PRGE", "start": 0, "end": 0}],
                  "text": "p53 and MK2",
                  "title": ""}

    return annotation

def test_ground_no_entites(blank_annotation):
    """Asserts that `grounding_utils.ground()` returns the expected value for a simple example with
    no identified entities.
    """

    actual = grounding_utils.ground(blank_annotation)
    expected = blank_annotation

    assert actual == expected

def test_ground_chemicals(ched_annotation):
    """Asserts that `grounding_utils.ground()` returns the expected value for a simple example with
    chemical entities.
    """
    glucose_xrefs = [
        {'namespace': 'TODO', 'id': 'CIDs00005793'},
        {'namespace': 'TODO', 'id': 'CIDs10954115'},
        {'namespace': 'TODO', 'id': 'CIDs53782692'},
    ]
    fructose_xrefs = [{'namespace': 'TODO', 'id': 'CIDs00439709'}]

    ched_annotation['ents'][0].update(xrefs=glucose_xrefs)
    ched_annotation['ents'][1].update(xrefs=fructose_xrefs)

    actual = grounding_utils.ground(ched_annotation)
    expected = ched_annotation

    assert actual == expected

def test_ground_diso(diso_annotation):
    """Asserts that `grounding_utils.ground()` returns the expected value for a simple example with
    disease entities.
    """
    cancer_xrefs = [{'namespace': 'TODO', 'id': 'DOID:162'}]
    cystic_fibrosis_xrefs = [{'namespace': 'TODO', 'id': 'DOID:1485'}]

    diso_annotation['ents'][0].update(xrefs=cancer_xrefs)
    diso_annotation['ents'][1].update(xrefs=cystic_fibrosis_xrefs)

    actual = grounding_utils.ground(diso_annotation)
    expected = diso_annotation

    assert actual == expected

def test_ground_livb(livb_annotation):
    """Asserts that `grounding_utils.ground()` returns the expected value for a simple example with
    species entities.
    """
    mouse_xrefs = [
        {'namespace': 'TODO', 'id': '10090'},
        {'namespace': 'TODO', 'id': '10088'},
    ]
    human_xrefs = [{'namespace': 'TODO', 'id': '9606'}]

    livb_annotation['ents'][0].update(xrefs=mouse_xrefs)
    livb_annotation['ents'][1].update(xrefs=human_xrefs)

    actual = grounding_utils.ground(livb_annotation)
    expected = livb_annotation

    assert actual == expected

def test_ground_prge(prge_annotation):
    """Asserts that `grounding_utils.ground()` returns the expected value for a simple example with
    species entities.
    """
    p53_xrefs = [{'namespace': 'TODO', 'id': 'ENSP00000269305', 'organism-id': '9606'}]
    mk2_xrefs = [
        {'namespace': 'TODO', 'id': 'ENSP00000356070', 'organism-id': '9606'},
        {'namespace': 'TODO', 'id': 'ENSP00000433109', 'organism-id': '9606'},
    ]

    prge_annotation['ents'][0].update(xrefs=p53_xrefs)
    prge_annotation['ents'][1].update(xrefs=mk2_xrefs)

    actual = grounding_utils.ground(prge_annotation)
    expected = prge_annotation

    assert actual == expected
