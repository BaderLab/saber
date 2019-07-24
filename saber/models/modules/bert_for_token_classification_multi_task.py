import logging

from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert import BertModel, BertForTokenClassification

logger = logging.getLogger(__name__)


class BertForTokenClassificationMultiTask(BertForTokenClassification):
    """Multi-task BERT model for token-level classification.

    This module is composed of the BERT model with multiple linear layers on top of the full hidden
    state of the last layer, one per 'task' (dataset).

    Attributes:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the
            scripts `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with
            the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and
            type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with
            indices selected in [0, 1]. It's a mask to be used if the input sequence length is
            smaller than the max input sequence length in the current batch. It's the mask that we
            typically use for attention when a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size,
            sequence_length] with indices selected in [0, ..., num_labels].
        `model_idx`: # TODO (John).

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

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
    def __init__(self, config, num_labels):
        # TODO (John): Once the latest release of pytorch-pretrained-bert is released you can
        # remove `num_labels[0]`
        super(BertForTokenClassificationMultiTask, self).__init__(config, num_labels[0])

        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList(
            [nn.Linear(config.hidden_size, labels) for labels in num_labels]
        )
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                model_idx=-1):
        sequence_output, _ = self.bert(input_ids=input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask,
                                       output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        # Access classifier corresponding to the model for this dataset
        logits = self.classifier[model_idx](sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels[model_idx])[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels.long())
            else:
                loss = loss_fct(logits.view(-1, self.num_labels[model_idx]), labels.view(-1))
            return loss
        else:
            return logits
