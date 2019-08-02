import copy
from itertools import permutations

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

# TODO (John): This can be shortned to from pytorch_transformers import x after next release
from pytorch_transformers.modeling_bert import BertPreTrainedModel
from pytorch_transformers import BertModel

from ...constants import CHUNK_END_TAGS
from ...constants import NEG_VALUE
from ...constants import TOK_MAP_PAD
from .biaffine_classifier import BiaffineAttention


class BertForEntityAndRelationExtraction(BertPreTrainedModel):
    """A BERT based model for joint named entity recognition (NER) and relation extraction (RE).

    Arguments:
        config (BertConfig): `BertConfig` class instance with a configuration to build a new model.

    Inputs:
        input_ids (torch.LongTensor): Of shape `(batch_size, sequence_length)`, indices of input
            sequence tokens in the vocabulary. To match pre-training, BERT input sequence should be
            formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:
                tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1

            (b) For single sequences:
                tokens:         [CLS] the dog is hairy . [SEP]
                token_type_ids:   0   0   0   0  0     0   0

            Indices can be obtained using `pytorch_transformers.BertTokenizer`. See
            `pytorch_transformers.PreTrainedTokenizer.encode` and
            `pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        orig_to_tok_map (torch.LongTensor): Of shape `[batch_size, sequence_length]` which
            contains the original tokens positions before WordPiece tokenization was applied.
        token_type_ids (torch.LongTensor): Optional, of shape `(batch_size, sequence_length)`.
            Segment token indices to indicate first and second portions of the inputs. Indices are
            selected in `[0, 1]`: `0` corresponds to a `sentence A` token, `1` corresponds to a
            `sentence B` token (see `BERT: Pre-training of Deep Bidirectional Transformers for
            Language Understanding`_ for more details).
        attention_mask (torch.LongTensor): Optional, of shape `(batch_size, sequence_length)`.
            Mask to avoid performing attention on padding token indices. Mask values selected in
            `[0, 1]`: 1` for tokens that are NOT MASKED, `0` for MASKED tokens.
        labels (torch.LongTensor): Optional, of shape `(batch_size, sequence_length)` containing
            labels for computing the token classification loss. Indices should be in
            `[0, ..., config.num_labels]`.
        ent_labels (torch.LongTensor): Optional, of shape `(batch_size, sequence_length)` containing
            labels for computing the token classification loss. Indices should be in
            `[0, ..., config.num_labels]`.
        rel_labels: TODO.
        position_ids (torch.LongTensor): Optional, of shape `(batch_size, sequence_length)`. Indices
            of positions of each input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        head_mask (torch.FloatTensor): Optional, of shape `(num_heads,)` or
            `(num_layers, num_heads)`: Mask to nullify selected heads of the self-attention modules.
            Mask values selected in `[0, 1]`: `1` indicates the head is not masked, `0` indicates
            the head is masked.

    # TODO (John): This is in flux, update it when we settle on an output structure.
    Outputs (Tuple): Comprising various elements depending on the configuration (`config`) and inputs:
        loss (torch.FloatTensor): Optional, returned when `labels` is provided, of shape `(1,)`.
            Classification loss.
        scores (torch.FloatTensor): Of shape `(batch_size, sequence_length, config.num_labels)`
            Classification scores (before SoftMax).
        hidden_states (list): Optional, returned when `config.output_hidden_states=True`. `list` of
            `torch.FloatTensor` (one for the output of each layer + the output of the embeddings) of
            shape (batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the
            output of each layer plus the initial embedding outputs.
        attentions (list): Optional, returned when config.output_attentions=True`. `list` of
            torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after
            the attention softmax, used to compute the weighted average in the self-attention heads.

    # TODO (John): This is in flux, update it when we settle on an output structure.
    Example:

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertForEntityAndRelationExtraction(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids, labels=labels)
        >>> loss, scores = outputs[:2]
    """
    def __init__(self, config):
        super(BertForEntityAndRelationExtraction, self).__init__(config)
        # TODO (John): Eventually support MTL.
        self.idx_to_ent = self.config.idx_to_ent[0]
        self.num_ent_labels = self.config.num_ent_labels[0]
        self.num_rel_labels = self.config.num_rel_labels[0]

        self.ent_class_weights = self.config.__dict__.get('ent_class_weights')
        self.rel_class_weights = self.config.__dict__.get('rel_class_weights')

        # NER Module
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ent_classifier = nn.Linear(config.hidden_size, self.num_ent_labels)

        self.apply(self.init_weights)

        # TODO (John): Once I settle on some kind of structure of hyperparams (a dict?) place
        # these there.
        # entity_embed_size = self.saber_config.entity_embed_size
        # head_tail_ffnns_size = self.saber_config.head_tail_ffnns_size
        entity_embed_size = 128
        head_tail_ffnns_size = 512

        # RE module
        self.embed = nn.Embedding(self.num_ent_labels, entity_embed_size)

        # encoder_layer = nn.TransformerEncoderLayer(config.hidden_size + entity_embed_size, 2)
        # encoder_norm = nn.LayerNorm(config.hidden_size + entity_embed_size)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2, encoder_norm)

        # MLPs for head and tail projections
        projection = nn.Sequential(
            nn.Linear(config.hidden_size + entity_embed_size, head_tail_ffnns_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(head_tail_ffnns_size, head_tail_ffnns_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)

        self.rel_classifier = BiaffineAttention(head_tail_ffnns_size // 2, self.num_rel_labels)

    def forward(self, input_ids, orig_to_tok_map, token_type_ids=None, attention_mask=None,
                ent_labels=None, rel_labels=None, position_ids=None, head_mask=None):
        # Forward pass through BERT
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

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
        # sequence_output = self.transformer_encoder(sequence_output,
        #                                            src_key_padding_mask=attention_mask)

        with torch.no_grad():
            ent_indices, proj_rel_labels = self._get_entities_and_labels(
                orig_to_tok_map=orig_to_tok_map,
                ner_preds=ner_preds,
                idx_to_ent=self.idx_to_ent,
                ent_labels=ent_labels,
                rel_labels=rel_labels
            )

            ent_indices = ent_indices.to(input_ids.device)
            proj_rel_labels = proj_rel_labels.to(input_ids.device)

        if ent_indices.nelement() > 0:
            heads = sequence_output[ent_indices[:, 0], ent_indices[:, 1], :]
            tails = sequence_output[ent_indices[:, 0], ent_indices[:, 2], :]

            # Project entity pairs into head/tail space
            heads = self.ffnn_head(heads)
            tails = self.ffnn_tail(tails)

            # RE classification
            re_logits = self.rel_classifier(heads, tails)

        else:
            ent_indices = None
            re_logits = None

        # Add hidden states and attention if they are present
        outputs = (ner_logits, re_logits, ) + outputs[2:]
        if ent_labels is not None and rel_labels is not None:
            # TODO (John): Need better API for this.
            ent_class_weights = (torch.tensor(self.ent_class_weights).to(input_ids) if
                                 self.ent_class_weights is not None else None)
            rel_class_weights = (torch.tensor(self.rel_class_weights).to(input_ids) if
                                 self.rel_class_weights is not None else None)

            loss_fct_ner = CrossEntropyLoss(weight=ent_class_weights)
            loss_fct_re = CrossEntropyLoss(weight=rel_class_weights)

            # Computing NER loss
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ner_logits.view(-1, self.num_ent_labels)[active_loss]
                active_labels = ent_labels.view(-1)[active_loss]
                ner_loss = loss_fct_ner(active_logits, active_labels.long())
            else:
                ner_loss = \
                    loss_fct_ner(ner_logits.view(-1, self.num_ent_labels), ent_labels.view(-1))

            # Computing RE loss
            # If no relations were predicted, we assign 0 vector for each true relation (this
            # represents a maximally confused classifier.)
            if re_logits is None:
                re_logits = torch.zeros((proj_rel_labels.size(0), self.num_rel_labels),
                                        device=input_ids.device)
            # Otherwise, we need to add 0 vector for each missed relation
            else:
                n_missing_logits = proj_rel_labels.size(0) - re_logits.size(0)
                if n_missing_logits > 0:
                    missing_logits = torch.zeros((n_missing_logits, self.num_rel_labels),
                                                 device=input_ids.device)
                    re_logits = torch.cat((re_logits, missing_logits))

            re_loss = loss_fct_re(re_logits.view(-1, self.num_rel_labels), proj_rel_labels.view(-1))

            outputs = (ner_loss, re_loss, proj_rel_labels, ner_logits, re_logits) + outputs[2:]

        # (ner_loss, re_loss, proj_rel_labels), ner_logits, re_logits, (hidden_states), (attentions)
        return outputs

    # TODO (John): This is a disaster. It works, but I should find a way to clean it up and drop
    # as many explict for loops as possible.
    def _get_entities_and_labels(self,
                                 orig_to_tok_map,
                                 ner_preds,
                                 idx_to_ent,
                                 ent_labels=None,
                                 rel_labels=None):
        """Returns the indices of all predicted relation heads/tails and corresponding gold labels.
        """
        ent_indices, proj_rel_labels, missed_rel_labels = [], [], []

        for sent_idx, (tok_map, ner_pred) in enumerate(zip(orig_to_tok_map, ner_preds)):
            tok_map, ner_pred = orig_to_tok_map[sent_idx], ner_preds[sent_idx]
            # Get all indices representing original tokens that were predicted to be a
            # standalone entity or the end of a entity chunk
            ent_idxs = \
                [tok_idx.item() for tok_idx in tok_map if tok_idx != TOK_MAP_PAD and
                 idx_to_ent[ner_pred[tok_idx].item()].startswith(tuple(CHUNK_END_TAGS))]

            # Permutate these predicted entities to get all candidate entities
            ent_idxs = list(permutations(ent_idxs, r=2))

            # Add a sentence index to the predicted relation
            if ent_idxs:
                ent_indices.extend([[sent_idx] + list(idx) for idx in ent_idxs])

            # Project gold labels onto predicted labels
            if ent_labels is not None and rel_labels is not None:
                # Construct a map of heads: tails: relation type
                rel_lab_map = {}
                for rel in rel_labels[sent_idx]:
                    head, tail, label = tok_map[rel[0]].item(), tok_map[rel[1]].item(), rel[2]
                    if head in rel_lab_map:
                        rel_lab_map[head][tail] = label
                    else:
                        rel_lab_map[head] = {tail: label}

                # Get gold labels for predicted relations
                for head, tail in ent_idxs:
                    try:
                        correct_entities = (ent_labels[sent_idx][head] == ner_pred[head] and
                                            ent_labels[sent_idx][tail] == ner_pred[tail])
                        if correct_entities:
                            proj_rel_labels.append(rel_lab_map[head][tail])
                            del rel_lab_map[head][tail]
                        # If the entities are not correct, label the relation NEG
                        else:
                            proj_rel_labels.append(NEG_VALUE)
                    except KeyError:
                        proj_rel_labels.append(NEG_VALUE)

                # Get labels of missed relations
                for tail in rel_lab_map.values():
                    missed_rel_labels.extend(list(tail.values()))

        # Convert everything to tensors
        ent_indices = torch.as_tensor(ent_indices, dtype=torch.long)

        proj_rel_labels += missed_rel_labels
        if proj_rel_labels or rel_labels is None:
            proj_rel_labels = torch.as_tensor(proj_rel_labels, dtype=torch.long)
        else:
            proj_rel_labels = torch.zeros(1, dtype=torch.long)

        return ent_indices, proj_rel_labels
