"""Test suite for the `app_utils` module (saber.utils.app_utils).
"""
import pytest

from ..utils import app_utils
from .resources.constants import DUMMY_ENTITIES


class TestAppUtils(object):
    """Collects all unit tests for `saber.cli.app`.
    """
    def test_get_pubmed_xml_errors(self):
        """Asserts that call to `app_utils.get_pubmed_xml()` raises a ValueError error when an invalid
        value for argument `pmid` is passed."""
        invalid_pmids = [["test"], "test", 0.0, 0, -1, (42,)]

        for pmid in invalid_pmids:
            with pytest.raises(ValueError):
                app_utils.get_pubmed_xml(pmid)

    def test_harmonize_entities(self):
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
