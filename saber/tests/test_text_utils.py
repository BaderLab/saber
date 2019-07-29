"""Test suite for the `text_utils` module (saber.utils.text_utils).
"""


class TestTextUtils(object):
    """Collects all unit tests for `saber.utils.bert_utils`.
    """
    def test_biomedical_tokenizer(self, nlp_with_biomedical_tokenizer):
        """Asserts that call to customized spaCy tokenizer returns the expected results.
        """
        # the empty string
        blank_text = ""
        blank_expected = []
        # simple test with no complexities
        simple_text = "This is an easy test."
        simple_expected = ["This", "is", "an", "easy", "test", "."]
        # complicated test with some important edge cases
        complicated_test = ("This test's tokenizers handeling of very-tricky situations, 3X, "
                            "more/or/less.")
        complicated_expected = ["This", "test", "'", "s", "tokenizers", "handeling", "of",
                                "very", "-", "tricky", "situations", ",", "3X", ",", "more", "/",
                                "or", "/", "less", "."]

        # these tests were taken straight from training data
        from_CHED_ds = ("The results have shown that the degradation product p-choloroaniline is"
                        " not a significant factor in chlorhexidine-digluconate associated erosive "
                        "cystitis.")
        from_CHED_ds_expected = ['The', 'results', 'have', 'shown', 'that', 'the', 'degradation',
                                 'product', 'p', '-', 'choloroaniline', 'is', 'not', 'a',
                                 'significant', 'factor', 'in', 'chlorhexidine', '-', 'digluconate',
                                 'associated', 'erosive', 'cystitis', '.']
        from_DISO_ds = ("Rats were treated with seven day intravenous infusion of fucoidan "
                        "(30 micrograms h-1) or vehicle.")
        from_DISO_expected = ['Rats', 'were', 'treated', 'with', 'seven', 'day', 'intravenous',
                              'infusion', 'of', 'fucoidan', '(', '30', 'micrograms', 'h', '-', '1',
                              ')', 'or', 'vehicle', '.']
        from_LIVB_ds = ("Methanoregula formicica sp. nov., a methane-producing archaeon isolated"
                        " from methanogenic sludge.")
        from_LIVB_ds_expected = ['Methanoregula', 'formicica', 'sp', '.', 'nov', '.', ',', 'a',
                                 'methane', '-', 'producing', 'archaeon', 'isolated', 'from',
                                 'methanogenic', 'sludge', '.']
        from_PRGE_ds = ("Here we report the cloning, expression, and biochemical characterization"
                        " of the 32-kDa subunit of human (h) TFIID, termed hTAFII32.")
        from_PRGE_ds_expected = ['Here', 'we', 'report', 'the', 'cloning', ',', 'expression', ',',
                                 'and', 'biochemical', 'characterization', 'of', 'the', '32', '-',
                                 'kDa', 'subunit', 'of', 'human', '(', 'h', ')', 'TFIID', ',',
                                 'termed', 'hTAFII32', '.']

        # generic tests
        assert [t.text for t in nlp_with_biomedical_tokenizer(blank_text)] == blank_expected
        assert [t.text for t in nlp_with_biomedical_tokenizer(simple_text)] == simple_expected
        assert [t.text for t in nlp_with_biomedical_tokenizer(complicated_test)] == \
            complicated_expected
        # tests taken straight from training data
        assert [t.text for t in nlp_with_biomedical_tokenizer(from_CHED_ds)] == \
            from_CHED_ds_expected
        assert [t.text for t in nlp_with_biomedical_tokenizer(from_DISO_ds)] == \
            from_DISO_expected
        assert [t.text for t in nlp_with_biomedical_tokenizer(from_LIVB_ds)] == \
            from_LIVB_ds_expected
        assert [t.text for t in nlp_with_biomedical_tokenizer(from_PRGE_ds)] == \
            from_PRGE_ds_expected
