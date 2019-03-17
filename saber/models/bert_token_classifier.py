"""Contains the BertForTokenClassification PyTorch model for squence labelling.
"""
import logging
import shutil

import numpy as np
import torch
from pytorch_pretrained_bert import (BertConfig, BertForTokenClassification,
                                     BertTokenizer)
from tqdm import tqdm

from .. import constants
from ..metrics import Metrics
from ..utils import data_utils, model_utils
from .base_model import BasePyTorchModel

LOGGER = logging.getLogger(__name__)

# TODO (johngiorgi): Write train_valid_test and cross-validation as local functions of `train()`

class BertTokenClassifier(BasePyTorchModel):
    """A PyTorch implementation of a BERT model for sequence labeling.

    A BERT for token classification model implemented in PyTorch.

    Args:
        config (Config): Contains a set of harmonzied arguments provided in a *.ini file and,
            optionally, from the command line.
        datasets (list): A list of Dataset objects.
        pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint

    References:
        - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://arxiv.org/abs/1810.04805
        - PyTorch Implementation of BERT by Huggingface: https://github.com/huggingface/pytorch-pretrained-BERT
    """
    def __init__(self, config, datasets, pretrained_model_name_or_path='bert-base-uncased', **kwargs):
        super().__init__(config, datasets, **kwargs)
        # by default, place the model on the CPU
        self.device = torch.device("cpu")
        # number of GPUs available
        self.n_gpus = 0
        # the name or path of a pre-trained BERT model
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # a tokenizer which corresponds to the pre-trained model to load
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name_or_path,
                                                       do_lower_case=False)

    def load(self, model_filepath):
        """Load a model from disk.

        Loads a PyTorch model from disk by loading its architecture and weights from a `.bin` file
        at `model_filepath`.

        Args:
            model_filepath (str): filepath to the models architecture (`.bin` file).
        """
        # TODO (johngiorgi): Test that saving loading from CPU/GPU works as expected
        model_state_dict = torch.load(model_filepath, map_location=lambda storage, loc: storage)
        # this is a trick to get the number of labels
        num_labels = len(model_state_dict['classifier.bias'])
        # TODO (johngiorgi): Can we get the model name from the model_state_dict?
        model = BertForTokenClassification.from_pretrained(self.pretrained_model_name_or_path,
                                                           num_labels=num_labels,
                                                           state_dict=model_state_dict)
        # get the device the model will live on, along with number of GPUs available
        model, self.device, self.n_gpus = model_utils.get_device(model)

        self.models.append(model)

    def specify(self):
        """Specifies an op-for-op PyTorch implementation of Google's BERT for sequence tagging.

        Specifies and initilizaties an op-for-op PyTorch implementation of Google's BERT. Model
        is appended to `self.models`.
        """
        # (TODO): In future, it would be nice to support MTL. For now, initializing
        # BertTokenClassifier with multiple datasets will simply create an independent model for
        # for each dataset.
        for dataset in self.datasets:
            # plus 1 is necessary to account for 'X' tag introduced by workpiece tokenization
            num_labels = len(dataset.type_to_idx['tag']) + 1
            model = BertForTokenClassification.from_pretrained(self.pretrained_model_name_or_path,
                                                               num_labels=num_labels)
            # get the device the model will live on, along with number of GPUs available
            model, self.device, self.n_gpus = model_utils.get_device(model)

            self.models.append(model)

    def prepare_data_for_training(self):
        """Returns a list containing the training data for each dataset at `self.datasets`.

        For each dataset at `self.datasets`, collects the data to be used for training.
        Each dataset is represented by a dictionary, where the keys 'x_<partition>' and
        'y_<partition>' contain the inputs and targets for each partition 'train', 'valid', and
        'test'.
        """
        # (TEMP): In future, would be nice to support mt learning, hence why training_data is a list
        training_data = []
        for ds in self.datasets:
            # type to idx must be modified to be compatible with BERT model
            model_utils.setup_type_to_idx_for_bert(ds)
            # collect data for each partition (train, valid, test) in a dataset
            data = {}
            for partition in constants.PARTITIONS:
                # some partitions (valid, test) may not be provided by user
                if ds.type_seq[partition] is not None:
                    x_idx, y_idx, attention_masks = \
                        model_utils.process_data_for_bert(tokenizer=self.tokenizer,
                                                          word_seq=ds.type_seq[partition]['word'],
                                                          tag_seq=ds.type_seq[partition]['tag'],
                                                          tag_to_idx=ds.type_to_idx['tag'])
                    data['x_{}'.format(partition)] = [x_idx, attention_masks]
                    data['y_{}'.format(partition)] = y_idx
                else:
                    data['x_{}'.format(partition)] = None
                    data['y_{}'.format(partition)] = None

            training_data.append(data)

        return training_data

    def train_valid_test(self):
        """Trains a PyTorch model with a cross-validation strategy.

        Trains a PyTorch model (`self.model.models`) or models in the case of multi-task learning
        (`self.model.models` is a list of PyTorch models) using a cross-validation strategy. Expects
        only a train partition to have been supplied.
        """
        width, _ = shutil.get_terminal_size()
        print('Using train/test/valid strategy...')
        print('\n{}\n'.format('-' * width))
        LOGGER.info('Using a train/test/valid strategy for training')

        # TODO (johngiorgi): In future, we will support MTL, hence why these are lists
        # get the train / valid partitioned data for all datasets and all folds
        training_data = self.prepare_data_for_training()

        output_dir = model_utils.prepare_output_directory(self.config)[0]

        model = self.models[0]
        dataset = self.datasets[0]
        optimizer = model_utils.get_optimizers(self.models, self.config)[0]

        # use 10% of train data as validation data if no validation data provided
        if training_data['x_valid'] is None:
            training_data = data_utils.collect_valid_data(training_data)[0]

        # metrics object
        metrics = Metrics(config=self.config,
                          training_data=training_data,
                          idx_to_tag=dataset.idx_to_tag,
                          output_dir=output_dir,
                          model=self,
                          model_idx=0)

        # append dataloaders to training_data
        training_data['train_dataloader'] = \
            model_utils.get_dataloader_for_ber(x=training_data['x_train'][0],
                                               y=training_data['y_train'],
                                               attention_mask=training_data['x_train'][-1],
                                               config=self.config,
                                               data_partition='train')
        training_data['valid_dataloader'] = \
            model_utils.get_dataloader_for_ber(x=training_data['x_valid'][0],
                                               y=training_data['y_valid'],
                                               attention_mask=training_data['x_valid'][-1],
                                               config=self.config,
                                               data_partition='eval')

        training_data['test_dataloader'] = \
            model_utils.get_dataloader_for_ber(x=training_data['x_test'][0],
                                               y=training_data['y_test'],
                                               attention_mask=training_data['x_test'][-1],
                                               config=self.config,
                                               data_partition='eval')

        for epoch in range(self.config.epochs):
            model.train()
            tr_loss = 0
            nb_tr_steps = 0

            # setup a progress bar
            pbar_descr = 'Epoch: {}/{}'.format(epoch + 1, self.config.epochs)
            pbar = tqdm(training_data['train_dataloader'], unit='batch', desc=pbar_descr)

            for _, batch in enumerate(pbar):

                model.zero_grad()

                # update train loss in progress bar
                train_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0.
                pbar.set_postfix(train_loss=train_loss)

                # add batch to device
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # forward pass
                loss = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
                # backward pass
                loss.backward()

                # track train loss
                tr_loss += loss.item()
                nb_tr_steps += 1

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                               max_norm=self.config.grad_norm)
                # update parameters
                optimizer.step()

            # need to feed epoch argument manually, as this is a keras callback object
            metrics.on_epoch_end(epoch=metrics.current_epoch)

            pbar.close()

    def cross_validation(self):
        """Trains a PyTorch model with a cross-validation strategy.

        Trains a PyTorch model (`self.model.models`) or models in the case of multi-task learning
        (`self.model.models` is a list of PyTorch models) using a cross-validation strategy. Expects
        only a train partition to have been supplied.
        """
        width, _ = shutil.get_terminal_size()
        print('Using {}-fold cross-validation strategy.'.format(self.config.k_folds))
        print('\n{}\n'.format('-' * width))
        LOGGER.info('Using a %s-fold cross-validation strategy for training', self.config.k_folds)

        # get the train / valid partitioned data for all datasets and all folds
        training_data = self.prepare_data_for_training()
        training_data = data_utils.collect_cv_data(training_data, self.config.k_folds)[0]

        output_dir = model_utils.prepare_output_directory(self.config)[0]

        # TRAIN loop
        for fold in range(self.config.k_folds):
            # TODO (johngiorgi): In future, we will support MTL, hence why these are lists
            model = self.models[0]
            dataset = self.datasets[0]
            optimizer = model_utils.get_optimizers(self.models, self.config)[0]

            # append dataloaders to training_data
            training_data[fold]['train_dataloader'] = \
                model_utils.get_dataloader_for_ber(x=training_data[fold]['x_train'][0],
                                                   y=training_data[fold]['y_train'],
                                                   attention_mask=training_data[fold]['x_train'][-1],
                                                   config=self.config,
                                                   data_partition='train')
            training_data[fold]['valid_dataloader'] = \
                model_utils.get_dataloader_for_ber(x=training_data[fold]['x_valid'][0],
                                                   y=training_data[fold]['y_valid'],
                                                   attention_mask=training_data[fold]['x_valid'][-1],
                                                   config=self.config,
                                                   data_partition='eval')

            # metrics object
            metrics = Metrics(config=self.config,
                              training_data=training_data[fold],
                              idx_to_tag=dataset.idx_to_tag,
                              output_dir=output_dir,
                              model=self,
                              model_idx=0,
                              fold=fold)

            # get optimizers for each model
            for epoch in range(self.config.epochs):
                model.train()
                tr_loss = 0
                nb_tr_steps = 0

                # setup a progress bar
                fold_and_epoch = (fold + 1, self.config.k_folds, epoch + 1, self.config.epochs)
                pbar_descr = 'Fold: {}/{}, Epoch: {}/{}'.format(*fold_and_epoch)
                pbar = tqdm(training_data[fold]['train_dataloader'], unit='batch', desc=pbar_descr)

                for _, batch in enumerate(pbar):

                    model.zero_grad()

                    # update train loss in progress bar
                    train_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0.
                    pbar.set_postfix(train_loss=train_loss)

                    # add batch to gpu
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch

                    # forward pass
                    loss = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

                    # backward pass
                    loss.backward()

                    # loss object is a vector of size n_gpus, need to average if more than 1
                    if self.n_gpus > 1:
                        loss = loss.mean()

                    # track train loss
                    tr_loss += loss.item()
                    nb_tr_steps += 1

                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                   max_norm=self.config.grad_norm)
                    # update parameters
                    optimizer.step()

                # need to feed epoch argument manually, as this is a keras callback object
                metrics.on_epoch_end(epoch=metrics.current_epoch)

                pbar.close()

            # clear and rebuild the model at end of each fold (except for the last fold)
            if fold < self.config.k_folds - 1:
                self.reset_model()

    def eval(self, training_data, model_idx=0, partition='train'):
        """Get `y_true` and `y_pred` for given inputs and targets in `training_data`.

        Performs prediction for the model at `self.models[model_idx]`), and returns a 2-tuple
        containing the true (gold) labels and the predicted labels, where labels are integers
        corresponding to mapping at `self.idx_to_tag`. Inputs and labels are stored in a PyTorch
        Dataloader at `training_data[partition_dataloader]`.

        Args:
            training_data (dict): Contains the data (at key `partition_dataloader`).
            partition (str): Which partition to perform a prediction step on, must be one of
                'train', 'valid', 'test'.

        Returns:
            A two-tuple containing the gold label integer sequences and the predicted integer label
            sequences.
        """
        model = self.models[model_idx], dataset = self.datasets[model_idx]
        model.eval()  # puts the model in evaluation mode

        y_pred, y_true = [], []
        eval_loss, nb_eval_steps = 0, 0

        # get the dataloader for the given partition
        dataloader = training_data['{}_dataloader'.format(partition)]

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
                logits = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask)

                # loss object is a vector of size n_gpus, need to average if more than 1
                if self.n_gpus > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()

            # put the true labels & logits on the cpu and convert to numpy array
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            y_true.append(label_ids)
            y_pred.extend([list(p) for p in np.argmax(logits, axis=2)])

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

        # flatten arrays
        y_true = [[l_ii for l_ii in l_i] for l in y_true for l_i in l]
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # mask out pads and wordpiece tokens
        y_true, y_pred = model_utils.mask_labels(y_true, y_pred, dataset.type_to_idx['tag'][constants.PAD])
        y_true, y_pred = model_utils.mask_labels(y_true, y_pred, dataset.type_to_idx['tag'][constants.WORDPIECE])

        # sanity check
        if not y_true.shape == y_pred.shape:
            err_msg = ("'y_true' and 'y_pred' in 'BertForTokenClassification.eval() have different"
                       " shapes ({} and {} respectively)".format(y_true.shape, y_pred.shape))
            LOGGER.error('AssertionError: %s', err_msg)
            raise AssertionError(err_msg)

        return y_true, y_pred

    def predict(self, sents, model_idx=0):
        """Perform inference on tokenized and sentence segmented text.

        Using the model at `self.models[model_idx]`, performs inference on `sents`, a list of lists
        containing tokenized sentences. The result of this inference is a list of lists, where the
        outer lists correspond to sentences in `sents` and inner lists contains predicted indices
        for the corresponding tokens in `sents`.

        Args:
            sents (list): List of lists containing tokenized sentences to annotate.
            model_idx (int): Index to model in `self.models` that will be used for inference.
                Defaults to 0.
        """
        model = self.models[model_idx]
        model.eval()  # puts the model in evaluation mode

        # process the sentences into lists of lists of ids and corresponding attention masks
        X, _, attention_masks = model_utils.process_data_for_bert(self.tokenizer, sents)
        X, attention_masks = X.to(self.device), attention_masks.to(self.device)

        # actual prediction happens here
        with torch.no_grad():
            logits = model(X, token_type_ids=None, attention_mask=attention_masks)
        logits = logits.detach().cpu().numpy()
        X, y_pred = X.numpy(), np.asarray([list(pred) for pred in np.argmax(logits, axis=2)])

        # sanity check
        if not X.shape == y_pred.shape:
            err_msg = ("'X' and 'y_pred' in 'BertForTokenClassification.predict() have different"
                       " shapes ({} and {} respectively)".format(X.shape, y_pred.shape))
            LOGGER.error('AssertionError: %s', err_msg)
            raise AssertionError(err_msg)

        return X, y_pred
