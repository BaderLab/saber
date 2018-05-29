import utils_app

ENTITIES = {'ANAT': False,
            'CHED': True,
            'DISO': False,
            'LIVB': True,
            'PRGE': True,
            'TRIG': False
            }

def test_harmonize_entities():
    """Asserts that utils_web_service.harmonize_entities() returns the expected
    results."""
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
        utils_app.harmonize_entities(ENTITIES, one_on_test)
    assert multi_on_expected == \
        utils_app.harmonize_entities(ENTITIES, multi_on_test)
    assert none_on_expected == \
        utils_app.harmonize_entities(ENTITIES, none_on_test)
