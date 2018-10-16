"""Any and all unit tests for the app_utils (saber/utils/app_utils.py).
"""
import pytest

from ..utils import app_utils
from .resources.dummy_constants import *

############################################ UNIT TESTS ############################################

def test_get_pubmed_xml_errors():
    """Asserts that call to `app_utils.get_pubmed_xml()` raises a ValueError error when an invalid
    value for argument `pmid` is passed."""
    invalid_pmids = [["test"], "test", 0.0, 0, -1, (42,)]

    for pmid in invalid_pmids:
        with pytest.raises(ValueError):
            app_utils.get_pubmed_xml(pmid)

def test_harmonize_entities():
    """Asserts that app_utils.harmonize_entities() returns the expected results."""
    # single bool test
    one_on_test = {'PRGE': True}
    one_on_expected = {'ANAT': False, 'CHED': False, 'DISO': False,
                       'LIVB': False, 'PRGE': True, 'TRIG': False}
    # multi bool test
    multi_on_test = {'PRGE': True, 'CHED': True, 'TRIG': False}
    multi_on_expected = {'ANAT': False, 'CHED': True, 'DISO': False,
                         'LIVB': False, 'PRGE': True, 'TRIG': False}
    # null test
    none_on_test = {}
    none_on_expected = {'ANAT': False, 'CHED': False, 'DISO': False,
                        'LIVB': False, 'PRGE': False, 'TRIG': False}

    assert one_on_expected == \
        app_utils.harmonize_entities(DUMMY_ENTITIES, one_on_test)
    assert multi_on_expected == \
        app_utils.harmonize_entities(DUMMY_ENTITIES, multi_on_test)
    assert none_on_expected == \
        app_utils.harmonize_entities(DUMMY_ENTITIES, none_on_test)
