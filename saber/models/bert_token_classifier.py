"""Contains the BertForTokenClassification PyTorch model for squence labelling.
"""
import logging
from itertools import chain

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from pytorch_pretrained_bert import (BertAdam, BertConfig,
                                     BertForTokenClassification, BertTokenizer)
from torch.optim import Adam
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from .. import constants
from ..metrics import Metrics
from ..preprocessor import Preprocessor
from ..utils import generic_utils
from .base_model import BasePyTorchModel

LOGGER = logging.getLogger(__name__)

# TODO (johnmgiorgi): This should be handeled better. Maybe as a config argument.
PYTORCH_BERT_MODEL = 'bert-base-cased'

class BertTokenClassifier(BasePyTorchModel):
    """A PyTorch implementation of a BERT model for sequence labeling.

    A BERT for token classification model implemented in PyTorch.

    Args:
        config (Config): Contains a set of harmonzied arguments provided in a *.ini file and,
            optionally, from the command line.
        datasets (list): A list of Dataset objects.

    References:
        - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://arxiv.org/abs/1810.04805
        - PyTorch Implementation of BERT by Huggingface: https://github.com/huggingface/pytorch-pretrained-BERT
    """
    def __init__(self, config, datasets, **kwargs):
        super().__init__(config, datasets, **kwargs)

    def load(self, weights_filepath, model_filepath):
        """Load a model from disk.

        Loads a PyTorch model from disk.
        """
        # TODO (James): Is there anything specific about the loading of THIS PyTorch model?
        # if so, overwrite BasePyTorchModel.load(), if not, delete this entirely.
        pass

    def specify(self):
        # plus 1 is necc to account for 'X' tag introduced by workpiece tokenization
        num_labels = len(self.datasets[0].type_to_idx['tag']) + 1
        model = BertForTokenClassification.from_pretrained(PYTORCH_BERT_MODEL, num_labels=num_labels)
        self.models.append(model)

    def prepare_data_for_training(self):
        """Returns a list containing the training data for each dataset at `self.datasets`.

        For each dataset at `self.datasets`, collects the data to be used for training.
        Each dataset is represented by a dictionary, where the keys 'x_<partition>' and
        'y_<partition>' contain the inputs and targets for each partition 'train', 'valid', and
        'test'.
        """
        tokenizer = BertTokenizer.from_pretrained(PYTORCH_BERT_MODEL, do_lower_case=False)

        # (TEMP): In future, would be nice to support mt learning, hence why training_data is a list
        training_data = []
        for ds in self.datasets:
            # necc for dataset to be compatible with BERTS wordpiece tokenization
            ds.type_to_idx['tag']['X'] = len(ds.type_to_idx['tag'])
            ds.get_idx_to_tag()

            # collect train data, must be provided
            x_train_type, y_train_type = \
                self.tokenize_for_bert(tokenizer, ds.type_seq['train']['word'], ds.type_seq['train']['tag'])
            x_train_idx, y_train_idx, attention_mask_train = \
                self.type_to_idx_for_bert(tokenizer, x_train_type, y_train_type, ds.type_to_idx['tag'])


            attention_mask_train, attention_mask_valid, _, _ = train_test_split(attention_mask_train, x_train_idx, random_state=2018, test_size=0.1)
            x_train_idx, x_valid_idx, y_train_idx, y_valid_idx = train_test_split(x_train_idx, y_train_idx, random_state=2018, test_size=0.1)


            train_dataloader = self.get_dataloader_for_ber(x_train_idx, y_train_idx, attention_mask_train, data_partition='train')
            valid_dataloader = self.get_dataloader_for_ber(x_valid_idx, y_valid_idx, attention_mask_valid, data_partition='eval')

            '''
            # collect valid and test data, may not be provided
            valid_dataloader, test_dataloader = None, None
            if ds.idx_seq['valid'] is not None:
                x_valid_type, y_valid_type = \
                    self.tokenize_for_bert(tokenizer, ds.type_seq['valid']['word'], ds.type_seq['valid']['tag'])
                x_valid_idx, y_valid_idx, attention_mask_valid = \
                    self.type_to_idx_for_bert(tokenizer, x_valid_type, y_valid_type, ds.type_to_idx['tag'])
                valid_dataloader = self.get_dataloader_for_ber(x_valid_idx, y_valid_idx, attention_mask_valid, data_partition='eval')
            if ds.idx_seq['test'] is not None:
                x_test_type, y_test_type = \
                    self.tokenize_for_bert(tokenizer, ds.type_seq['test']['word'], ds.type_seq['test']['tag'])
                x_test_idx, y_test_idx, attention_mask_test = \
                    self.type_to_idx_for_bert(tokenizer, x_test_type, y_test_type, ds.type_to_idx['tag'])
                test_dataloader = self.get_dataloader_for_ber(x_test_idx, y_test_idx, attention_mask_test, data_partition='eval')
            '''

            training_data.append({'train': train_dataloader,
                                  'valid': valid_dataloader,
                                  # 'test': test_dataloader,
                                 })

        return training_data


    def tokenize_for_bert(self, tokenizer, word_seq, tag_seq):
        """
        """
        # produces a list of list of lists, with the innermost list containing wordpieces
        bert_tokenized_text = [[tokenizer.wordpiece_tokenizer.tokenize(token) for token in sent] for
                               sent in word_seq]
        # recreate the label sequence by adding special 'X' tag for wordpieces
        bert_tokenized_labels = []

        for i, _ in enumerate(bert_tokenized_text):
            tokenized_labels = []
            for token, label in zip(bert_tokenized_text[i], tag_seq[i]):
                tokenized_labels.append([label] + ['X'] * (len(token) - 1))
            bert_tokenized_labels.append(list(chain.from_iterable(tokenized_labels)))

        # flatten the tokenized text back to a list of lists
        bert_tokenized_text = [list(chain.from_iterable(sent)) for sent in bert_tokenized_text]

        # check that we didn't screw anything up
        assert np.asarray(bert_tokenized_text).shape == np.asarray(bert_tokenized_labels).shape

        return bert_tokenized_text, bert_tokenized_labels

    def type_to_idx_for_bert(self, tokenizer, word_seq, tag_seq, tag_to_idx):
        """
        """
        # process words
        bert_word_indices = [tokenizer.convert_tokens_to_ids(sent) for sent in word_seq]
        bert_word_indices = pad_sequences(bert_word_indices,
                                          maxlen=constants.MAX_SENT_LEN,
                                          dtype="long",
                                          truncating="post",
                                          padding="post")
        # process tags
        bert_tag_indices = [[tag_to_idx.get(tag, constants.UNK) for tag in sent] for sent in tag_seq]
        bert_tag_indices = pad_sequences(bert_tag_indices,
                                         maxlen=constants.MAX_SENT_LEN,
                                         value=tag_to_idx[constants.PAD],
                                         padding="post",
                                         dtype="long",
                                         truncating="post")

        # generate attention masks for padded data
        bert_attention_masks = [[float(idx > 0) for idx in sent] for sent in bert_word_indices]

        from prettytable import PrettyTable

        table = PrettyTable()

        table.field_names = ['Input ID', 'Reverse Input ID', 'Label ID', 'Reverse Label ID']

        for input_, tag in zip(bert_word_indices[0], bert_tag_indices[0]):
            row = [input_, tokenizer.ids_to_tokens[input_], tag, self.datasets[0].idx_to_tag[tag]]
            table.add_row(row)

        # print the first 30 rows from the first input example as a sanity check
        print(table[:30])

        return bert_word_indices, bert_tag_indices, bert_attention_masks

    def get_dataloader_for_ber(self, x, y, attention_mask, data_partition='train'):
        """
        """
        if data_partition not in {'train', 'eval'}:
            err_msg = ("Expected one of 'train', 'eval' for argument `data_partition` to "
                       "`get_dataloader_for_ber`. Got: '{}'".format(data_partition))
            LOGGER.error('ValueError %s', err_msg)
            raise ValueError(err_msg)

        x = torch.tensor(x)
        y = torch.tensor(y)
        attention_mask = torch.tensor(attention_mask)

        if data_partition == 'train':
            data = TensorDataset(x, attention_mask, y)
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.config.batch_size)
        elif data_partition == 'eval':
            data = TensorDataset(x, attention_mask, y)
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.config.batch_size)

        return dataloader

    def train(self):
        """
        """
        # (TODO): In future, would be nice to support mt learning, hence why these are all lists
        model = self.models[0]
        dataset = self.datasets[0]
        training_data = self.prepare_data_for_training()[0]

        # use a GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()

            model.cuda()

            print('Using CUDA(s) device with name: {}'.format(torch.cuda.get_device_name(0)))
        else:
            device = torch.device("cpu")
            print('No GPU available for training. Using CPU')

        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        optimizer = Adam(optimizer_grouped_parameters, lr=self.config.learning_rate)

        for _ in range(self.config.epochs):
            # TRAIN loop
            model.train()
            tr_loss = 0
            nb_tr_steps = 0
            # position 0 is only to make the progress bar work on Colab, delete when we move to script
            for _ , batch in enumerate(tqdm(training_data['train'], desc='Epoch {}'.format(nb_tr_steps + 1))):
                model.zero_grad()
                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # forward pass
                loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.config.grad_norm)
                # update parameters
                optimizer.step()
            # print train loss per epoch
            print("Train loss: {}".format(tr_loss / nb_tr_steps))
            # VALIDATION on validation set
            self.evaluate(model, training_data, device, dataset.idx_to_tag)

    def evaluate(self, model, dataloaders, device, idx_to_tag):
        """
        """
        model.eval()  # puts the model in evaluation mode

        for partition in ['train', 'valid']:
            predictions, true_labels = [], []
            eval_loss, nb_eval_steps = 0, 0
            dataloader = dataloaders[partition]

            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    tmp_eval_loss = model(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)
                    logits = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask)

                # put the true labels & logits on the cpu and convert to numpy array
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                true_labels.append(label_ids)
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

            # flatten 3D array to 2D array
            # TODO (johnmgiorgi): Can I use chain.iterable for this?
            labels = [[l_ii for l_ii in l_i] for l in true_labels for l_i in l]

            y_true_tag = [idx_to_tag[idx] for idx in np.asarray(labels).ravel()]
            y_pred_tag = [idx_to_tag[idx] for idx in np.asarray(predictions).ravel()]

            # Use Saber to chunk entities and compute performance metrics
            y_true_chunks = Preprocessor.chunk_entities(y_true_tag)
            y_pred_chunks = Preprocessor.chunk_entities(y_pred_tag)

            results = Metrics.get_precision_recall_f1_support(y_true=y_true_chunks,
                                                              y_pred=y_pred_chunks,
                                                              criteria='exact')
            print()
            Metrics.print_performance_scores(results, title=partition)
            print("Valid/Test loss: {}".format(eval_loss / nb_eval_steps))
