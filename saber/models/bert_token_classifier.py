"""Contains the BertForTokenClassification PyTorch model for squence labelling.
"""
import logging

import numpy as np
import torch
from pytorch_pretrained_bert import (BertConfig, BertForTokenClassification, BertTokenizer)
from tqdm import tqdm

from .. import constants
from ..metrics import Metrics
from ..preprocessor import Preprocessor
from ..utils import data_utils, model_utils
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

    def load(self, model_filepath):
        """Load a model from disk.

        Loads a PyTorch model from disk by loading its architecture and weights from a `.bin` file
        at `model_filepath`.

        Args:
            model_filepath (str): filepath to the models architecture (`.bin` file).
        """
        # TODO (James): Fill this in based on your stuff in the notebook
        # TODO (James): In the future, we would like to support MTL. So self.models is a list.
        ### YOUR CODE STARTS HERE ####
        # model_state_dict = torch.load(output_model_file)
        # num_labels = len(model_state_dict['classifier.bias'])
        # model = BertForTokenClassification.from_pretrained(PYTORCH_BERT_MODEL,
        #                                                    num_labels=num_labels,
        #                                                    state_dict=model_state_dict)
        ### YOUR CODE ENDS HERE ####
        # self.models.append(model)
        pass

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
            model = BertForTokenClassification.from_pretrained(PYTORCH_BERT_MODEL,
                                                               num_labels=num_labels)
            # Get the device the model will live on
            self.device = model_utils.get_device(model)

            self.models.append(model)

    def prepare_data_for_training(self):
        """Returns a list containing the training data for each dataset at `self.datasets`.

        For each dataset at `self.datasets`, collects the data to be used for training.
        Each dataset is represented by a dictionary, where the keys 'x_<partition>' and
        'y_<partition>' contain the inputs and targets for each partition 'train', 'valid', and
        'test'.
        """
        bert_tokenizer = BertTokenizer.from_pretrained(PYTORCH_BERT_MODEL, do_lower_case=False)
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
                        model_utils.process_data_for_bert(tokenizer=bert_tokenizer,
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

    def train(self):
        """Trains a PyTorch model with a cross-validation strategy.

        Trains a PyTorch model (`self.model.models`) or models in the case of multi-task learning
        (`self.model.models` is a list of PyTorch models) using a cross-validation strategy. Expects
        only a train partition to have been supplied.
        """
        print('Using {}-fold cross-validation strategy...'.format(self.config.k_folds))
        LOGGER.info('Using a %s-fold cross-validation strategy for training', self.config.k_folds)

        # get the train/valid partitioned data for all datasets and all folds
        training_data = self.prepare_data_for_training()
        training_data = data_utils.collect_cv_data(training_data, self.config.k_folds)[0]

        output_dir = model_utils.prepare_output_directory(self.config)[0]

        for fold in range(self.config.k_folds):
            # in the future, it would be nice to support MTL, hence why these are lists
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
                              fold=fold)

            # get optimizers for each model
            for epoch in range(self.config.epochs):
                train_info = (fold + 1, self.config.k_folds, epoch + 1, self.config.epochs)
                print('Fold: {}/{}; Global epoch: {}/{}\n{}'.format(*train_info, '-' * 30))

                # get the device model will train on
                device = model_utils.get_device(model)
                # TRAIN loop
                model.train()
                tr_loss = 0
                nb_tr_steps = 0
                for _ , batch in enumerate(tqdm(training_data[fold]['train_dataloader'], desc='Epoch {}'.format(nb_tr_steps + 1))):
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
                # computes metrics, saves to disk
                # need to feed epoch argument manually, as this is a keras callback object
                metrics.on_epoch_end(epoch=metrics.current_epoch)

            # clear and rebuild the model at end of each fold (except for the last fold)
            if fold < self.config.k_folds - 1:
                self.reset_model()

    def prediction_step(self, training_data, partition):
        """
        """
        # TEMP: This will cause MT to break, but should work for now
        # TEMP: In future the model or its index will need to be passed to this function
        model = self.models[0]
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

        return y_true, y_pred
