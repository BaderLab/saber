import logging

from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_transformers.modeling_bert import BertPreTrainedModel
from pytorch_transformers import BertModel

logger = logging.getLogger(__name__)


class BertForTokenClassificationMultiTask(BertPreTrainedModel):
    """Multi-task BERT model for multi-task named entity recognition (NER).

    Attributes:
         config (pytorch_transformers.BertConfig): Model configuration class with all the
            hyperparameters of the model.

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
        position_ids (torch.LongTensor): Optional, of shape `(batch_size, sequence_length)`. Indices
            of positions of each input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        head_mask (torch.FloatTensor): Optional, of shape `(num_heads,)` or
            `(num_layers, num_heads)`: Mask to nullify selected heads of the self-attention modules.
            Mask values selected in `[0, 1]`: `1` indicates the head is not masked, `0` indicates
            the head is masked.
        model_idx (int): Optional, an index into `self.classifier` corresponding to which classifier
                to use in the case of a multi-task model. Defaults to -1, which will use the last
                classifier in `self.classifier`.

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

    Example:

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertForTokenClassificationMultiTask(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)
        >>> labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids, labels=labels)
        >>> loss, scores = outputs[:2]
    """
    def __init__(self, config):
        super(BertForTokenClassificationMultiTask, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList(
            [nn.Linear(config.hidden_size, nl) for nl in self.num_labels]
        )

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, model_idx=-1):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # Access classifier corresponding to the model for this dataset
        logits = self.classifier[model_idx](sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels[model_idx])[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels[model_idx]), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
