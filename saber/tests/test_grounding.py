import pytest

from ..utils import grounding

@pytest.fixture
def annotate():
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

def test_uniprot_search():
    annotation = annotate()
    print(str(annotation))

    assert len(grounding.uniprot_search("MK2"))==0

#