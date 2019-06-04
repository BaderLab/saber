"""Contains the BertForNER PyTorch model for named entity recognition with BERT.
"""
import logging

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer
from .bert_for_token_classification_multi_task import BertForTokenClassificationMultiTask
from tqdm import tqdm

from .. import constants
from ..utils import data_utils, model_utils
from .base_model import BasePyTorchModel
from itertools import zip_longest

LOGGER = logging.getLogger(__name__)

# TODO (johngiorgi): Test that saving loading from CPU/GPU works as expected


class BertForNER(BasePyTorchModel):
    """A PyTorch implementation of a BERT model for named entity recognition.

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
    def __init__(self, config, datasets, pretrained_model_name_or_path='bert-base-cased', **kwargs):
        super().__init__(config, datasets, **kwargs)
        # Place the model on the CPU by default
        self.device = torch.device("cpu")
        self.n_gpus = 0

        # Name or path of a pre-trained BERT model
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        self.num_labels = [len(dataset.idx_to_tag) + 1 for dataset in self.datasets]

        self.model_name = 'bert-ner'

    def load(self, model_filepath):
        """Loads a PyTorch BERT model for named entity recognition from disk.

        Loads a PyTorch BERT model for named entity recognition from disk and its corresponding
        tokenizer (saved with `BertForNER.save()`) by loading its architecture and weights from a
        `.bin` file at `model_filepath`.

        Args:
            model_filepath (str): filepath to the models architecture (`.bin` file).

        Returns:
            The loaded PyTorch BERT model and the corresponding Tokenizer.
        """
        model_state_dict = torch.load(model_filepath, map_location=lambda storage, loc: storage)

        # A trick to get the number of labels for each classifier
        i, num_labels = 0, []
        while f'classifier.{i}.bias' in model_state_dict:
            num_labels.append(len(model_state_dict[f'classifier.{i}.bias']))
            i += 1

        model = \
            BertForTokenClassificationMultiTask.from_pretrained(self.pretrained_model_name_or_path,
                                                                num_labels=num_labels,
                                                                state_dict=model_state_dict)
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name_or_path,
                                                  do_lower_case=False)

        # Get the device the model will live on, along with number of GPUs available
        self.device, self.n_gpus = model_utils.get_device(model)

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def specify(self):
        """Specifies a PyTorch BERT model for named entity recognition and its tokenizer.

        Returns:
            The loaded PyTorch BERT model and the corresponding Tokenizer.
        """
        if self.pretrained_model_name_or_path in constants.PRETRAINED_MODELS:
            self.pretrained_model_name_or_path = \
                model_utils.download_model_from_gdrive(self.pretrained_model_name_or_path)

        model = \
            BertForTokenClassificationMultiTask.from_pretrained(self.pretrained_model_name_or_path,
                                                                num_labels=self.num_labels)
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name_or_path,
                                                  do_lower_case=False)

        # Get the device the model will live on, along with number of GPUs available
        self.device, self.n_gpus = model_utils.get_device(model)

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def prepare_data_for_training(self):
        """Returns a list containing the training data for each dataset at `self.datasets`.

        For each dataset at `self.datasets`, collects the data to be used for training.
        Each dataset is represented by a dictionary, where the keys 'x_<partition>' and
        'y_<partition>' contain the inputs and targets for each partition 'train', 'valid', and
        'test'.
        """
        training_data = []
        for ds in self.datasets:
            # Type to idx must be modified to be compatible with BERT model
            model_utils.setup_type_to_idx_for_bert(ds)
            # Collect data for each partition (train, valid, test) in a dataset
            data = {}

            for partition in constants.PARTITIONS:
                # Some partitions (valid, test) may not be provided by user
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

    def train(self):
        """Co-ordinates the training of the PyTorch model `self.model`.

        Co-ordinates the training of the PyTorch model `self.model`. Minimally expects a train
        partition and one or both of valid and test partitions to be supplied in the Dataset objects
        at `self.datasets`.
        """
        # Gather everything we need to run a training session
        optimizers = self.prepare_optimizers()
        training_data = self.prepare_data_for_training()
        output_dir = model_utils.prepare_output_directory(self.config)

        def train_valid_test(training_data, output_dir, optimizers):
            # use 10% of train data as validation data if no validation data provided
            if training_data[0]['x_valid'] is None:
                training_data = data_utils.collect_valid_data(training_data)

            # get list of Keras Callback objects for computing/storing metrics
            metrics = model_utils.setup_metrics_callback(model=self,
                                                         config=self.config,
                                                         datasets=self.datasets,
                                                         training_data=training_data,
                                                         output_dir=output_dir,)

            # TODO (John): Dataloaders should be handled outside of the train loop
            for model_idx, train_data in enumerate(training_data):
                # append dataloaders to training_data
                train_data['train_dataloader'] = \
                    model_utils.get_dataloader_for_bert(
                        x=train_data['x_train'][0],
                        y=train_data['y_train'],
                        attention_mask=train_data['x_train'][-1],
                        model_idx=np.tile(model_idx, (train_data['x_train'][0].shape[0], 1)),
                        batch_size=self.config.batch_size,
                        data_partition='train')
                train_data['valid_dataloader'] = \
                    model_utils.get_dataloader_for_bert(
                        x=train_data['x_valid'][0],
                        y=train_data['y_valid'],
                        attention_mask=train_data['x_valid'][-1],
                        model_idx=np.tile(model_idx, (train_data['x_valid'][0].shape[0], 1)),
                        batch_size=self.config.batch_size,
                        data_partition='eval')

                train_data['test_dataloader'] = \
                    model_utils.get_dataloader_for_bert(
                        x=train_data['x_test'][0],
                        y=train_data['y_test'],
                        attention_mask=train_data['x_test'][-1],
                        model_idx=np.tile(model_idx, (train_data['x_test'][0].shape[0], 1)),
                        batch_size=self.config.batch_size,
                        data_partition='eval')

            # Collect dataloader, we don't need the other data
            dataloaders = [train_data['train_dataloader'] for train_data in training_data]
            total = max((len(dataloader)) for dataloader in dataloaders)

            for epoch in range(self.config.epochs):

                self.model.train()

                train_loss = [0] * len(self.num_labels)
                train_steps = [0] * len(self.num_labels)

                # Setup a progress bar
                pbar_descr = f'Epoch: {epoch + 1}/{self.config.epochs}'
                pbar = tqdm(zip_longest(*dataloaders),
                            unit='batch',
                            desc=pbar_descr,
                            total=total,
                            dynamic_ncols=True)

                for _, batches in enumerate(pbar):
                    for batch in batches:
                        if batch is not None:
                            # Add batch to device
                            batch = tuple(t.to(self.device) for t in batch)
                            input_ids, attention_mask, labels, model_idx = batch

                            model_idx = model_idx[0].item()

                            optimizer = optimizers[model_idx]
                            optimizer.zero_grad()

                            # Freeze any classifiers not currently being trained
                            for idx, _ in enumerate(self.num_labels):
                                self.model.classifier[idx].requires_grad = True \
                                    if model_idx == idx else False

                            loss = self.model(input_ids,
                                              token_type_ids=torch.zeros_like(input_ids),
                                              attention_mask=attention_mask,
                                              labels=labels,
                                              model_idx=model_idx)

                            loss.backward()

                            # Loss is a vector of size n_gpus, need to average if more than 1
                            if self.n_gpus > 1:
                                loss = loss.mean()

                            # Track train loss
                            train_loss[model_idx] += loss.item()
                            train_steps[model_idx] += 1

                            optimizer.step()

                            # Update train loss in progress bar
                            postfix = {
                                f'loss_{i}': f'{loss / steps:.4f}' if steps > 0 else 0.
                                for i, (loss, steps) in enumerate(zip(train_loss, train_steps))
                            }
                            pbar.set_postfix(postfix)

                for metric in metrics:
                    # Need to feed epoch argument manually, as this is a keras callback object
                    metric.on_epoch_end(epoch=epoch)

                pbar.close()

        def cross_validation(training_data, output_dir, optimizers):
            # Get the train / valid partitioned data for all datasets and all folds
            training_data = data_utils.collect_cv_data(training_data, self.config.k_folds)

            # Training loop
            for fold in range(self.config.k_folds):

                # Get list of Keras Callback objects for computing/storing metrics
                metrics = model_utils.setup_metrics_callback(model=self,
                                                             config=self.config,
                                                             datasets=self.datasets,
                                                             training_data=training_data,
                                                             output_dir=output_dir,
                                                             fold=fold)

                # TODO (John): Dataloaders should be handled outside of the train loop
                for model_idx, train_data in enumerate(training_data):
                    train_data[fold]['train_dataloader'] = \
                        model_utils.get_dataloader_for_bert(
                            x=train_data[fold]['x_train'][0],
                            y=train_data[fold]['y_train'],
                            attention_mask=train_data[fold]['x_train'][-1],
                            model_idx=np.tile(model_idx,
                                              (train_data[fold]['x_train'][0].shape[0], 1)),
                            batch_size=self.config.batch_size,
                            data_partition='train')
                    train_data[fold]['valid_dataloader'] = \
                        model_utils.get_dataloader_for_bert(
                            x=train_data[fold]['x_valid'][0],
                            y=train_data[fold]['y_valid'],
                            attention_mask=train_data[fold]['x_valid'][-1],
                            model_idx=np.tile(model_idx,
                                              (train_data[fold]['x_valid'][0].shape[0], 1)),
                            batch_size=self.config.batch_size,
                            data_partition='eval')

                # Collect dataloader, we don't need the other data
                dataloaders = [train_data[fold]['train_dataloader'] for train_data in training_data]
                total = max((len(dataloader)) for dataloader in dataloaders)

                for epoch in range(self.config.epochs):

                    self.model.train()

                    train_loss = [0] * len(self.num_labels)
                    train_steps = [0] * len(self.num_labels)

                    # Setup a progress bar
                    fold_and_epoch = (fold + 1, self.config.k_folds, epoch + 1, self.config.epochs)
                    pbar_descr = 'Fold: {}/{}, Epoch: {}/{}'.format(*fold_and_epoch)
                    pbar = tqdm(zip_longest(*dataloaders),
                                unit='batch',
                                desc=pbar_descr,
                                total=total,
                                dynamic_ncols=True)

                    for _, batches in enumerate(pbar):
                        for batch in batches:
                            if batch is not None:
                                batch = tuple(t.to(self.device) for t in batch)
                                input_ids, attention_mask, labels, model_idx = batch

                                model_idx = model_idx[0].item()

                                optimizer = optimizers[model_idx]
                                optimizer.zero_grad()

                                # Freeze any classifiers not currently being trained
                                for idx, _ in enumerate(self.num_labels):
                                    self.model.classifier[idx].requires_grad = True \
                                        if model_idx == idx else False

                                loss = self.model(input_ids,
                                                  token_type_ids=torch.zeros_like(input_ids),
                                                  attention_mask=attention_mask,
                                                  labels=labels,
                                                  model_idx=model_idx)

                                loss.backward()

                                # Loss is a vector of size n_gpus, need to average if more than 1
                                if self.n_gpus > 1:
                                    loss = loss.mean()

                                # Track train loss
                                train_loss[model_idx] += loss.item()
                                train_steps[model_idx] += 1

                                optimizer.step()

                            # Update train loss in progress bar
                            postfix = {
                                f'loss_{i}': f'{loss / steps:.4f}' if steps > 0 else 0.
                                for i, (loss, steps) in enumerate(zip(train_loss, train_steps))
                            }
                            pbar.set_postfix(postfix)

                    for metric in metrics:
                        # Need to feed epoch argument manually, as this is a keras callback object
                        metric.on_epoch_end(epoch=epoch)

                    pbar.close()

                # Clear and rebuild the model at end of each fold (except for the last fold)
                if fold < self.config.k_folds - 1:
                    self.reset_model()
                    optimizers = self.prepare_optimizers()

        # TODO: User should be allowed to overwrite this
        if training_data[0]['x_valid'] is not None or training_data[0]['x_test'] is not None:
            print('Using train/test/valid strategy...')
            LOGGER.info('Used a train/test/valid strategy for training')
            train_valid_test(training_data, output_dir, optimizers)
        else:
            print(f'Using {self.config.k_folds}-fold cross-validation strategy...')
            LOGGER.info('Used %s-fold cross-validation strategy for training', self.config.k_folds)
            cross_validation(training_data, output_dir, optimizers)

    def evaluate(self, training_data, model_idx=-1, partition='train'):
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
        self.model.eval()

        y_pred, y_true = [], []
        eval_loss, eval_steps = 0, 0

        # Get the dataset / dataloader for the given partition
        dataset = self.datasets[model_idx]
        dataloader = training_data[f'{partition}_dataloader']

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, labels, _ = batch

            with torch.no_grad():
                tmp_eval_loss = self.model(input_ids,
                                           token_type_ids=torch.zeros_like(input_ids),
                                           attention_mask=attention_mask,
                                           labels=labels,
                                           model_idx=model_idx)
                logits = self.model(input_ids,
                                    token_type_ids=torch.zeros_like(input_ids),
                                    attention_mask=attention_mask,
                                    model_idx=model_idx)

                # loss object is a vector of size n_gpus, need to average if more than 1
                if self.n_gpus > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()

            # put the true labels & logits on the cpu and convert to numpy array
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            y_true.append(label_ids)
            y_pred.extend([list(p) for p in np.argmax(logits, axis=2)])

            eval_loss += tmp_eval_loss.mean().item()
            eval_steps += 1

        # Flatten arrays
        y_true = [[l_ii for l_ii in l_i] for l in y_true for l_i in l]
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # Mask out pads and wordpiece tokens
        y_true, y_pred = model_utils.mask_labels(y_true, y_pred,
                                                 dataset.type_to_idx['tag'][constants.PAD])
        y_true, y_pred = model_utils.mask_labels(y_true, y_pred,
                                                 dataset.type_to_idx['tag'][constants.WORDPIECE])

        # Sanity check
        if not y_true.shape == y_pred.shape:
            err_msg = (f"'y_true' and 'y_pred' in 'BertForNER.evaluate() have"
                       " different shapes ({y_true.shape} and {y_pred.shape} respectively)")
            LOGGER.error('AssertionError: %s', err_msg)
            raise AssertionError(err_msg)

        return y_true, y_pred

    def predict(self, sents, model_idx=None):
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
        self.model.eval()

        # Process the sentences into lists of lists of ids and corresponding attention masks
        X, _, attention_masks = model_utils.process_data_for_bert(self.tokenizer, sents)

        X = torch.tensor(X).to(device=self.device)
        attention_masks = torch.tensor(attention_masks).to(device=self.device)

        # Actual prediction happens here
        with torch.no_grad():
            logits = self.model(X, token_type_ids=None, attention_mask=attention_masks)
        X, y_pred = X.detach().cpu().numpy(), logits.detach().cpu().numpy()

        if model_idx is not None:
            y_pred = y_pred[model_idx]

        return X, y_pred

    def prepare_optimizers(self):
        """Returns a list of PyTorch optimizers, one per output layer in `self.model`.

        For each output layer in `self.model`, creates an optmizer based on the given config at
        `self.config`. For a single-task model, the returned list will be of length 1. For
        a multi-task model, the returned list will be of length `len(self.model.ouput)`.

        Returns:
            A list of PyTorch optimizers initiated from the given config at `self.config`.
        """
        optimizers = [model_utils.get_bert_optimizer(self.model, self.config)
                      for _, _ in enumerate(self.num_labels)]

        return optimizers
