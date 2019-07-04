import logging
from itertools import permutations

import torch
from pytorch_pretrained_bert import BertForTokenClassification
from pytorch_pretrained_bert import BertModel
from torch import nn
from torch.nn import CrossEntropyLoss

from ... import constants
from ...constants import NEG_VALUE
from ...constants import TOK_MAP_PAD
from .biaffine_classifier import BiaffineAttention

logger = logging.getLogger(__name__)


class BertForJointEntityAndRelationClassification(BertForTokenClassification):
    """A BERT based model for joint entity and relation extraction.

    This module is composed of a named entity recognition (NER) module and relation extraction (RE)
    module. The NER module consists of a BERT model with a linear layer, which feeds predicted
    entities into the RE module, a transformer encoder block with a biaffine classifier.

    Arguments:
        config (BertConfig): `BertConfig` class instance with a configuration to build a new model.
        idx_to_ent (dict): A mapping of integers to their entity label (e.g. BIO, IOBES, etc.).
        num_ent_labels (int): the number of classes for the entity classifier. Defaults to `2`.
        num_rel_labels (int): the number of classes for the relation classifier. Defaults to `2`.

    Inputs:
        input_ids: A `torch.LongTensor` of shape `[batch_size, sequence_length]` with the word token
            indices in the vocabulary.
        orig_to_tok_map: A `torch.LongTensor` of shape `[batch_size, sequence_length]` which
            contains the original tokens positions before WordPiece tokenization was applied.
        token_type_ids: An optional `torch.LongTensor` of shape `[batch_size, sequence_length]`
            with the token types indices selected in `[0, 1]`. Type `0` corresponds to a
            `sentence A` and type `1` corresponds to a `sentence B` token (see BERT paper for more
            details).
        `attention_mask`: an optional `torch.LongTensor` of shape `[batch_size, sequence_length]`
            with indices selected in `[0, 1]`. It's a mask to be used if the input sequence length
            is smaller than the max input sequence length in the current batch. It's the mask that
            we typically use for attention when a batch has varying length sentences.
        `ent_labels`: Labels for the entity classification output: `torch.LongTensor` of shape
            `[batch_size, sequence_length]` with indices selected in `[0, ..., num_ent_labels]`.
        `rel_labels`: Labels for the relation classification output: `torch.LongTensor` of shape
            `[batch_size, sequence_length]` with indices selected in `[0, ..., num_rel_labels]`.

    # TODO (John): This is in flux, update it when we settle on an output structure.
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    # TODO (John): This is in flux, update it when we settle on an output structure.
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, idx_to_ent, num_ent_labels=2, num_rel_labels=2, **kwargs):
        super(BertForJointEntityAndRelationClassification, self).__init__(config, num_ent_labels[0])

        # TODO (John): Eventually support MTL. For now, assume training on self.datasets[0]
        self.idx_to_ent = idx_to_ent[0]
        self.num_ent_labels = num_ent_labels[0]
        self.num_rel_labels = num_rel_labels[0]

        # NER Module
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        del self.classifier  # Remove classifier the model comes with
        self.ent_classifier = nn.Linear(config.hidden_size, self.num_ent_labels)
        self.apply(self.init_bert_weights)

        # TODO (John): Once I settle on some kind of structure of hyperparams (a dict?) place
        # these there.
        entity_embed_size = 256
        head_tail_ffnns_size = 256

        # RC module
        self.embed = nn.Embedding(self.num_ent_labels, entity_embed_size)

        # Self-attention block
        encoder_layer = nn.TransformerEncoderLayer(config.hidden_size + entity_embed_size, 2)
        encoder_norm = nn.LayerNorm(config.hidden_size + entity_embed_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2, encoder_norm)

        # MLPs for head and tail projections
        self.ffnn_head = nn.Sequential(
            nn.Linear(config.hidden_size + entity_embed_size, head_tail_ffnns_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(head_tail_ffnns_size, head_tail_ffnns_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_tail = nn.Sequential(
            nn.Linear(config.hidden_size + entity_embed_size, head_tail_ffnns_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(head_tail_ffnns_size, head_tail_ffnns_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.rel_classifier = BiaffineAttention(head_tail_ffnns_size // 2, self.num_rel_labels)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self,
                input_ids,
                orig_to_tok_map,
                token_type_ids=None,
                attention_mask=None,
                ent_labels=None,
                rel_labels=None):

        # Forward pass through BERT
        sequence_output, _ = self.bert(input_ids=input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask,
                                       output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)

        # NER classification
        ner_logits = self.ent_classifier(sequence_output)

        # Map predicted NER labels to embeddings
        ner_preds = torch.argmax(ner_logits, dim=-1)
        embed_ent_labels = self.embed(ner_preds)

        # Concatenate output of BERT with embeddings of predicted token labels to get
        # entity aware contextualized word embeddings
        sequence_output = torch.cat((sequence_output, embed_ent_labels), dim=-1)

        # TODO (John): Need to add back in attention masks here
        # See https://github.com/pytorch/pytorch/issues/22374
        # sequence_output = self.transformer_encoder(sequence_output)

        with torch.no_grad():
            ent_indices, processed_rel_labels = self._get_entities_and_labels(sequence_output,
                                                                              ner_preds,
                                                                              orig_to_tok_map,
                                                                              rel_labels)
        if ent_indices:
            ent_indices = torch.cat(ent_indices)
            heads = sequence_output[ent_indices[:, 0], ent_indices[:, 1], :]
            tails = sequence_output[ent_indices[:, 0], ent_indices[:, 2], :]
            candidate_rels = torch.stack((heads, tails), dim=1)

            # Project entity pairs head/tail space
            head_proj = self.ffnn_head(candidate_rels[:, 0, :].squeeze(1))
            tail_proj = self.ffnn_tail(candidate_rels[:, -1, :].squeeze(1))

            # RE classification
            rc_logits = self.rel_classifier(head_proj, tail_proj)

        else:
            rc_logits = None

        if ent_labels is not None and rel_labels is not None:
            self.rel_class_weight = self.rel_class_weight.to(input_ids.device)
            loss_fct_ner = CrossEntropyLoss(weight=self.__dict__.get("ent_class_weight"))
            loss_fct_rc = CrossEntropyLoss(weight=self.rel_class_weight)
            # loss_fct_rc = CrossEntropyLoss(weight=self.__dict__.get("rel_class_weight"))

            # Computing NER loss
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ner_logits.view(-1, self.num_labels)[active_loss]
                active_labels = ent_labels.view(-1)[active_loss]
                loss_ner = loss_fct_ner(active_logits, active_labels.long())
            else:
                loss_ner = loss_fct_ner(ner_logits.view(-1, self.num_labels), ent_labels.view(-1))

            # Computing RE loss
            # If no relations were predicted, we assign 0 vector for each true relation (this
            # represents a maximially confused classifier.)
            if rc_logits is None:
                rc_logits = torch.zeros((processed_rel_labels.size(0), self.num_rel_labels),
                                        device=input_ids.device)
            # Otherwise, we need to add 0 vector for each missed relation
            else:
                n_missing_logits = processed_rel_labels.size(0) - rc_logits.size(0)
                if n_missing_logits > 0:
                    missing_logits = torch.zeros((n_missing_logits, self.num_rel_labels),
                                                 device=input_ids.device)
                    rc_logits = torch.cat((rc_logits, missing_logits))

            loss_re = loss_fct_rc(rc_logits.view(-1, self.num_rel_labels),
                                  processed_rel_labels.view(-1))

            return ner_logits, rc_logits, processed_rel_labels, loss_ner, loss_re
        else:
            return ner_logits, rc_logits

    # TODO (John): This is a disaster. It works, but I should find a way to clean it up and drop
    # as many explict for loops as possible.
    def _get_entities_and_labels(self,
                                 sequence_output,
                                 ner_preds,
                                 orig_to_tok_map,
                                 rel_labels=None):
        """Returns the indices of all predicted relation heads/tails and corresponding gold labels.
        """
        ent_indices, processed_rel_labels = [], []
        for i, _ in enumerate(sequence_output):
            tok_map, ner_pred = orig_to_tok_map[i], ner_preds[i]
            if rel_labels is not None:
                rel_labs = rel_labels[i]
            # Get all indices representing original tokens that were predicted to be a
            # standalone entity or the end of a entity chunk
            ent_idxs = \
                [idx.item() for idx in tok_map if idx != TOK_MAP_PAD and
                 # Check that this is the last token of a predicted span
                 self.idx_to_ent[ner_pred[idx].item()].startswith(tuple(constants.CHUNK_END_TAGS))]
            # permutate these predicted entities to get all candidate entities
            ent_idxs = list(permutations(ent_idxs, r=2))

            if rel_labels is not None:
                # Maps heads to tails to labels
                rel_lab_map = {
                    tok_map[lab[0]].item(): {tok_map[lab[1]].item(): lab[2]}
                    for lab in rel_labs if lab
                }

            if len(ent_idxs) > 1:
                ent_idxs = torch.as_tensor(ent_idxs, dtype=torch.long)
                sent_idxs = torch.zeros(ent_idxs.size(0), 1, dtype=torch.long).fill_(i)
                ent_idxs = torch.cat((sent_idxs, ent_idxs), dim=-1)
                ent_indices.append(ent_idxs)

                # Permutate these token positions to get all (unique) entity pairs
                if rel_labels is not None:
                    # Get labels for predicted relations
                    for _, head, tail in ent_idxs:
                        try:
                            processed_rel_labels.append(rel_lab_map[head.item()][tail.item()])
                            del rel_lab_map[head.item()][tail.item()]
                        except KeyError:
                            processed_rel_labels.append(NEG_VALUE)

                    # Get labels of missed relations
                    for tail in rel_lab_map.values():
                        processed_rel_labels.extend(list(tail.values()))

        if processed_rel_labels:
            processed_rel_labels = torch.as_tensor(processed_rel_labels,
                                                   device=sequence_output.device)
        else:
            processed_rel_labels = torch.as_tensor([0], device=sequence_output.device)

        return ent_indices, processed_rel_labels
