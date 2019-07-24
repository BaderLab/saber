from ..constants import TOK_MAP_PAD
from ..constants import OUTSIDE
from ..constants import PAD
from ..constants import PAD_VALUE
import torch

n_train_ex = 3
sent_len = 10
hs_size = 100

sequence_output = torch.randn(n_train_ex, sent_len, hs_size)
orig_to_tok_map = torch.as_tensor([
    [1, 2, 4, 5, 6, 7, 8, 9, TOK_MAP_PAD, TOK_MAP_PAD],
    [1, 2, 3, 5, 6, 7, 8, TOK_MAP_PAD, TOK_MAP_PAD, TOK_MAP_PAD],
    [1, 3, 4, 5, 6, 8, 9, TOK_MAP_PAD, TOK_MAP_PAD, TOK_MAP_PAD],
])
idx_to_ent = {
    PAD_VALUE: PAD,
    1: OUTSIDE,
    2: 'B-PRGE',
    3: 'E-PRGE',
    4: 'S-PRGE',
    # Mix tag styles so we can test both at once
    5: 'B-CHED',
    6: 'L-CHED',
    7: 'U-CHED',
}


class TestBertForJointEntityAndRelationExtraction(object):
    """Collects all unit tests for `saber.models.bert_for_joint_ner_and_rc.BertForJointNERAndRE`.
    """
    def test_get_entities_and_labels_no_pred_ents_no_rel_labels(self,
                                                                bert_for_joint_ner_and_rc_specify):
        """Assert that `BertForJointNERAndRE._get_entities_and_labels()` returns the expected
        results when there are no predited entities and no relation labels are provided.
        """
        # Collect dummy inputs to _get_entities_and_labels()
        ner_preds = torch.as_tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, PAD_VALUE, PAD_VALUE],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, PAD_VALUE, PAD_VALUE],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, PAD_VALUE, PAD_VALUE],
            ]
        )

        expected_ent_indices = torch.as_tensor([], dtype=torch.long)
        expected_proj_rel_labels = torch.as_tensor([], dtype=torch.long)

        actual_ent_indicies, actual_proj_rel_labels = \
            bert_for_joint_ner_and_rc_specify.model._get_entities_and_labels(
                orig_to_tok_map=orig_to_tok_map,
                ner_preds=ner_preds,
                idx_to_ent=idx_to_ent,
            )

        assert torch.equal(expected_ent_indices, actual_ent_indicies)
        assert torch.equal(expected_proj_rel_labels, actual_proj_rel_labels)

    def test_get_entities_and_labels_pred_ents_no_rel_labels(self,
                                                             bert_for_joint_ner_and_rc_specify):
        """Assert that `BertForJointNERAndRE._get_entities_and_labels()` returns the expected
        results when there are predited entities and no relation labels are provided.
        """
        ner_preds = torch.as_tensor(
            [
                [1, 1, 1, 2, 3, 1, 1, 1, 4, PAD_VALUE, PAD_VALUE],
                [1, 7, 1, 1, 1, 5, 6, 1, 1, PAD_VALUE, PAD_VALUE],
                [1, 1, 1, 7, 4, 1, 1, 5, 6, PAD_VALUE, PAD_VALUE],
            ]
        )

        # [sent_idx, head_idx, tail_idx]
        expected_ent_indices = torch.as_tensor([
            [0, 4, 8],
            [0, 8, 4],
            [1, 1, 6],
            [1, 6, 1],
            [2, 3, 4],
            [2, 3, 8],
            [2, 4, 3],
            [2, 4, 8],
            [2, 8, 3],
            [2, 8, 4],
        ], dtype=torch.long)
        expected_proj_rel_labels = torch.as_tensor([], dtype=torch.long)

        actual_ent_indicies, actual_proj_rel_labels = \
            bert_for_joint_ner_and_rc_specify.model._get_entities_and_labels(
                orig_to_tok_map=orig_to_tok_map,
                ner_preds=ner_preds,
                idx_to_ent=idx_to_ent,
            )

        assert torch.equal(expected_ent_indices, actual_ent_indicies)
        assert torch.equal(expected_proj_rel_labels, actual_proj_rel_labels)

    def test_get_entities_and_labels_no_pred_ents_with_rel_labels(self,
                                                                  bert_for_joint_ner_and_rc_specify):
        """Assert that `BertForJointNERAndRE._get_entities_and_labels()` returns the expected
        results when there are no predited entities and relation labels are provided.
        """
        ent_labels = torch.as_tensor(
            [
                [1, 1, 1, 2, 3, 1, 1, 1, 4, PAD_VALUE, PAD_VALUE],
                [1, 7, 1, 1, 1, 5, 6, 1, 1, PAD_VALUE, PAD_VALUE],
                [1, 1, 1, 7, 4, 1, 1, 5, 6, PAD_VALUE, PAD_VALUE],
            ]
        )
        ner_preds = torch.as_tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, PAD_VALUE, PAD_VALUE],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, PAD_VALUE, PAD_VALUE],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, PAD_VALUE, PAD_VALUE],
            ],
        )
        # [head_idx, tail_idx, rel_idx]
        rel_labels = [[[2, 6, 2]], [[4, 0, 3]], [[1, 2, 2], [1, 5, 2]]]

        expected_ent_indices = torch.as_tensor([], dtype=torch.long)
        expected_proj_rel_labels = torch.as_tensor([2, 3, 2, 2], dtype=torch.long)

        actual_ent_indicies, actual_proj_rel_labels = \
            bert_for_joint_ner_and_rc_specify.model._get_entities_and_labels(
                orig_to_tok_map=orig_to_tok_map,
                ner_preds=ner_preds,
                idx_to_ent=idx_to_ent,
                ent_labels=ent_labels,
                rel_labels=rel_labels,
            )

        assert torch.equal(expected_ent_indices, actual_ent_indicies)
        assert torch.equal(expected_proj_rel_labels, actual_proj_rel_labels)

    def test_get_entities_and_labels_pred_ents_with_rel_labels(self,
                                                               bert_for_joint_ner_and_rc_specify):
        """Assert that `BertForJointNERAndRE._get_entities_and_labels()` returns the expected
        results when there are predited entities and relation labels are provided.
        """
        ent_labels = torch.as_tensor(
            [
                [1, 1, 1, 2, 3, 1, 1, 1, 7, PAD_VALUE, PAD_VALUE],
                [1, 7, 1, 1, 1, 5, 6, 1, 1, PAD_VALUE, PAD_VALUE],
                [1, 1, 1, 7, 4, 1, 1, 5, 6, PAD_VALUE, PAD_VALUE],
            ]
        )
        ner_preds = torch.as_tensor(
            [
                [1, 1, 1, 2, 3, 1, 1, 1, 4, PAD_VALUE, PAD_VALUE],
                [1, 7, 1, 1, 1, 5, 6, 1, 1, PAD_VALUE, PAD_VALUE],
                [1, 1, 1, 7, 4, 1, 1, 5, 6, PAD_VALUE, PAD_VALUE],
            ]
        )
        rel_labels = [[[2, 6, 2]], [[4, 0, 3]], [[1, 2, 2], [1, 5, 2]]]

        # [sent_idx, head_idx, tail_idx]
        expected_ent_indices = torch.as_tensor([
            [0, 4, 8],
            [0, 8, 4],
            [1, 1, 6],
            [1, 6, 1],
            [2, 3, 4],
            [2, 3, 8],
            [2, 4, 3],
            [2, 4, 8],
            [2, 8, 3],
            [2, 8, 4],
        ], dtype=torch.long)
        expected_proj_rel_labels = torch.as_tensor([0, 0, 0, 3, 2, 2, 0, 0, 0, 0, 2],
                                                   dtype=torch.long)

        actual_ent_indicies, actual_proj_rel_labels = \
            bert_for_joint_ner_and_rc_specify.model._get_entities_and_labels(
                orig_to_tok_map=orig_to_tok_map,
                ner_preds=ner_preds,
                idx_to_ent=idx_to_ent,
                ent_labels=ent_labels,
                rel_labels=rel_labels,
            )

        assert torch.equal(expected_ent_indices, actual_ent_indicies)
        assert torch.equal(expected_proj_rel_labels, actual_proj_rel_labels)
