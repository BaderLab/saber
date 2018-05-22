import pytest

import utils_web_service

ENTITIES = {'ANAT': False,
            'CHED': False,
            'DISO': False,
            'LIVB': False,
            'PRGE': True,
            'TRIG': False
            }

def test_harmonize_entities():
    """Asserts that utils_web_service.harmonize_entities() returns the expected
    results."""
    # single bool test
    one_on_test = {'PRGE': True}
    one_on_expected = {'PRGE': True, 'LIVB': False, 'CHED': False,
                       'DISO': False, 'TRIG': False}
    # multi bool test
    multi_on_test = {'PRGE': True, 'CHED': True, 'TRIG': False}
    multi_on_expected = {'PRGE': True, 'LIVB': False, 'CHED': True,
                         'DISO': False, 'TRIG': False}
    # null test
    none_on_test = {}
    none_on_expected = {'PRGE': False, 'LIVB': False, 'CHED': False,
                        'DISO': False, 'TRIG': False}

    assert one_on_expected == \
        utils_web_service.harmonize_entities(ENTITIES, one_on_test)
    assert multi_on_expected == \
        utils_web_service.harmonize_entities(ENTITIES, multi_on_test)
    assert none_on_expected == \
        utils_web_service.harmonize_entities(ENTITIES, none_on_test)
