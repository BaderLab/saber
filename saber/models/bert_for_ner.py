import logging
from itertools import zip_longest

import torch
from pytorch_transformers import BertTokenizer
from tqdm import tqdm

from .. import constants
from ..constants import TOK_MAP_PAD
from ..constants import WORDPIECE
from ..utils import bert_utils
from ..utils import data_utils
from ..utils import model_utils
from ..utils.generic_utils import MissingStepException
from .base_model import BaseModel
from .modules.bert_for_token_classification_multi_task import \
    BertForTokenClassificationMultiTask

LOGGER = logging.getLogger(__name__)


class BertForNER(BaseModel):
    """A PyTorch implementation of a BERT model for named entity recognition (NER).

    Args:
        config (Config): Contains a set of harmonzied arguments provided in a *.ini file and,
            optionally, from the command line.
        datasets (list): A list of Dataset objects.
        pretrained_model_name_or_path: either:
                - a string with the `shortcut name` of a pre-trained model configuration to load
                    from cache or download and cache if not already stored in cache
                    (e.g. 'bert-base-uncased').
                - a path to a `directory` containing a configuration file saved
                    using the `save_pretrained(save_directory)` method.
                - a path or url to a saved configuration `file`.

    References:
        - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding:
            https://arxiv.org/abs/1810.04805
        - PyTorch Implementation of BERT by Huggingface:
            https://github.com/huggingface/pytorch-transformers
    """
    def __init__(self, config, datasets, pretrained_model_name_or_path='bert-base-cased'):
        super(BertForNER, self).__init__(config, datasets)

        # Get any CUDA devices that are available
        self.device, self.n_gpus = model_utils.get_device()

        self.num_labels = []
        for dataset in self.datasets:
            # Required for dataset to be compatible with BERTs wordpiece tokenization
            if WORDPIECE not in dataset.type_to_idx['ent']:
                dataset.type_to_idx['ent'][WORDPIECE] = len(dataset.type_to_idx['ent'])
                dataset.get_idx_to_tag()
            self.num_labels.append(len(dataset.idx_to_tag['ent']))

        # Name or path of a pre-trained PyTorch-Transformers BERT model
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.model_name = 'bert-ner'

    def save(self, directory):
        """Saves the `BertPreTrainedModel` model at `self.model`, its `BertConfig` and its
        `BertTokenizer` to disk."

        Args:
            directory (str): Directory to save the models weights, config and tokenizer.

        Returns:
            `directory`, the path to the directory where the model was saved.

        Raises:
            MissingStepException: If either `self.model` or `self.tokenizer` is None.
        """
        self.model, self.tokenizer = self.__dict__.get("model"), self.__dict__.get("tokenizer")

        if self.model is None or self.tokenizer is None:
            err_msg = ('self.model or self.tokenizer is None. Did you initialize a model'
                       ' before saving with BertForNER.specify() or BertForNER.load()?')
            LOGGER.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)

        # Saves model / config to directory/pytorch_model.bin / directory/config.json respectively
        self.model.save_pretrained(directory)
        self.tokenizer.save_vocabulary(directory)

        return directory

    def load(self, directory):
        """Loads a PyTorch `BertForTokenClassificationMultiTask` model from disk.

        Args:
            directory (str): Directory to save the models weights, config and tokenizer.

        Returns:
            The loaded PyTorch `BertPreTrainedModel` model and its corresponding `BertTokenizer`.
        """
        self.tokenizer = BertTokenizer.from_pretrained(directory)
        self.model = BertForTokenClassificationMultiTask.from_pretrained(directory).to(self.device)

        return self.model, self.tokenizer

    def specify(self):
        """Initializes a PyTorch `BertForTokenClassificationMultiTask` model and its tokenizer.

        Returns:
            The loaded PyTorch `BertPreTrainedModel` model and its corresponding `BertTokenizer`.
        """
        # If this is one of our preprocessed BERT models, download it from GDrive first
        if self.pretrained_model_name_or_path in constants.PRETRAINED_MODELS:
            self.pretrained_model_name_or_path = \
                model_utils.download_model_from_gdrive(self.pretrained_model_name_or_path)

        self.model = BertForTokenClassificationMultiTask.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            # Replace num_labels with our list
            num_labels=self.num_labels
        ).to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            do_lower_case=False
        )

        return self.model, self.tokenizer

    def prepare_data_for_training(self):
        """Returns a list containing data which has been processed for training with `self.model`.

        For each dataset at `self.datasets`, processes the data to be used for training and/or
        evaluation with `self.model`. Returns a list of dictionaries, of length `len(self.datasets)`
        containing a PyTorch Dataloader for each of a datasets partitions (e.g.
        `training_data[partition][0]['train']['dataloader']`) contains the Dataloader for the
        train partition of `self.datasets[0]`.

        Returns:
            A list of dictionaries, of length `len(self.datasets)`, containing a PyTorch Dataloader
            for each of a datasets partitions.
        """
        training_data = []
        for model_idx, dataset in enumerate(self.datasets):
            training_data.append([])  # for each dataset

            # Re-process the dataset to be compatible with BERT models
            processed_dataset = bert_utils.process_dataset_for_bert(dataset, self.tokenizer)
            # If no valid set provide, create one from the train set (or generate k-folds)
            processed_dataset = data_utils.prepare_data_for_eval(self.config, processed_dataset)

            # A hack to ensure that training data is always a list (datasets) of lists (folds)
            if not isinstance(processed_dataset, list):
                processed_dataset = [processed_dataset]

            for fold in processed_dataset:
                dataloaders = bert_utils.get_dataloader_for_bert(
                    processed_dataset=fold,
                    batch_size=self.config.batch_size,
                    model_idx=model_idx
                )

                # Add dataloader to the processed dataset and update training data
                for partition, dataloader in dataloaders.items():
                    if dataloader is not None:
                        fold[partition]['dataloader'] = dataloader

                training_data[-1].append(fold)

        return training_data

    def train(self):
        """Co-ordinates the training of the PyTorch model `self.model`.

        Co-ordinates the training of the PyTorch model `self.model`. Minimally expects a train
        partition to be supplied in the Dataset objects at `self.datasets`.
        """
        # Gather everything we need to run a training session
        optimizers = self.prepare_optimizers()
        training_data = self.prepare_data_for_training()
        output_dirs = model_utils.prepare_output_directory(self.config)

        metrics = model_utils.setup_metrics_callback(
            config=self.config,
            model=self,
            datasets=self.datasets,
            training_data=training_data,
            output_dirs=output_dirs,
        )

        # Training loop
        k_folds = len(training_data[0])
        for fold in range(k_folds):
            # Use mixed precision training if Apex is installed and CUDA device available
            try:
                if torch.cuda.is_available():
                    from apex import amp
                    self.model, optimizers = amp.initialize(self.model, optimizers, opt_level='O1')
                    LOGGER.info("Imported Apex. Training with mixed precision (opt_level='O1').")
                else:
                    LOGGER.info(('Apex is installed but no CUDA device is available. Training with'
                                 ' standard precision.'))
            except ImportError:
                print(("Install Apex for faster training times and reduced memory usage"
                       " (https://github.com/NVIDIA/apex)."))
                LOGGER.info("Couldn't import Apex. Training with standard precision.")

            # Use multiple-GPUS if available
            if self.n_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)

            # Collect dataloaders, we don't need the other data
            dataloaders = [data[fold]['train']['dataloader'] for data in training_data]
            total = len(max(dataloaders, key=len))

            for epoch in range(self.config.epochs):
                self.model.train()

                train_loss = [0] * len(self.num_labels)
                train_steps = [0] * len(self.num_labels)

                # Setup a progress bar
                if k_folds > 1:
                    fold_and_epoch = (fold + 1, k_folds, epoch + 1, self.config.epochs)
                    desc = 'Fold: {}/{}, Epoch: {}/{}'.format(*fold_and_epoch)
                else:
                    desc = f'Epoch: {epoch + 1}/{self.config.epochs}'

                pbar = tqdm(
                    zip_longest(*dataloaders),
                    unit='batch',
                    desc=desc,
                    total=total,
                    dynamic_ncols=True
                )

                for _, batches in enumerate(pbar):
                    for batch in batches:
                        if batch is not None:
                            _, input_ids, attention_mask, labels, _, model_idx = batch
                            model_idx = model_idx[0].item()

                            inputs = {
                                'input_ids': input_ids.to(self.device),
                                'token_type_ids': torch.zeros_like(input_ids).to(self.device),
                                'attention_mask': attention_mask.to(self.device),
                                'labels': labels.to(self.device),
                                'model_idx': model_idx,
                            }

                            optimizer = optimizers[model_idx]
                            optimizer.zero_grad()

                            outputs = self.model(**inputs)
                            loss = outputs[0]

                            # Loss is a vector of size n_gpus, need to average if more than 1
                            if self.n_gpus > 1:
                                loss = loss.mean()

                            try:
                                with amp.scale_loss(loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                               self.config.grad_norm)
                            except (ImportError, UnboundLocalError):
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.config.grad_norm)

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
                    metric.on_epoch_end()

                pbar.close()

            # Clear and rebuild the model at end of each fold (except for the last fold)
            if k_folds > 1 and fold < k_folds - 1:
                self.reset_model()
                optimizers = self.prepare_optimizers()

                for metric in metrics:
                    metric.on_fold_end()  # bumps internal fold counter

        return metrics

    def evaluate(self, training_data, partition='train'):
        """Perform a prediction step using `self.model` on `training_data[partition]`.

        Performs prediction for the model at `self.model` and returns a two-tuple containing the
        true (gold) NER labels and corresponding predicted labels.

        Args:
            training_data (dict): Contains the data (at key `[partition]['dataloader']`).
            partition (str): Which partition to perform a prediction step on, must be one of
                'train', 'valid', or 'test'.

        Returns:
            A two-tuple containing the gold NER label sequences and corresponding predicted labels.
        """
        self.model.eval()

        with torch.no_grad():

            y_true, y_pred = [], []
            eval_loss, eval_steps = 0, 0

            dataloader = training_data[partition]['dataloader']

            for batch in dataloader:
                _, input_ids, attention_mask, labels, orig_to_tok_map, model_idx = batch
                model_idx = model_idx[0].item()

                input_ids = input_ids.to(self.device)
                token_type_ids = torch.zeros_like(input_ids)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    model_idx=model_idx,
                )
                loss, logits = outputs[:2]

                preds = logits.argmax(dim=-1)

                # Loss object is a vector of size n_gpus, need to average if more than 1
                if self.n_gpus > 1:
                    loss = loss.mean()

                # TODO (John): There has got to be a better way to do this?
                # Mask [PAD] and WORDPIECE tokens
                for pred, labels, tok_map in zip(preds, labels, orig_to_tok_map):
                    orig_token_indices = torch.as_tensor(
                        [i for i in tok_map if i != TOK_MAP_PAD],
                        device=self.device,
                        dtype=torch.long
                    )

                    masked_labels = torch.index_select(labels, -1, orig_token_indices).tolist()
                    masked_preds = torch.index_select(pred, -1, orig_token_indices).tolist()

                    # Map predictions to tags
                    y_true.append([self.datasets[model_idx].idx_to_tag['ent'][idx]
                                  for idx in masked_labels])
                    y_pred.append([self.datasets[model_idx].idx_to_tag['ent'][idx]
                                  for idx in masked_preds])

                # TODO (John): We don't actually do anything with this?
                eval_loss += loss.item()
                eval_steps += 1

        return y_true, y_pred

    def predict(self, tokens):
        """Perform inference on tokenized and sentence segmented text.

        Using the model at `self.models[model_idx]`, performs inference on `sents`, a list of lists
        containing tokenized sentences. The result of this inference is a list of lists, where the
        outer lists correspond to sentences in `sents` and inner lists contains predicted indices
        for the corresponding tokens in `sents`.

        Args:
            tokens (list): List of lists containing tokenized sentences to annotate.

        Returns:
            If `self.models` has a single output layer:
                A list of lists, containing the predicted labels for `tokens`.
            If `self.models` has multiple output layers:
                A list of lists of lists, containing the predicted labels for `tokens` from each
                output layer in `self.models`.
        """
        self.model.eval()

        with torch.no_grad():
            # Prepare data for input to model
            bert_tokens, orig_to_tok_map = \
                bert_utils.wordpiece_tokenize_sents(tokens, tokenizer=self.tokenizer)
            input_ids, orig_to_tok_map, attention_mask = \
                bert_utils.index_pad_mask_bert_tokens(bert_tokens,
                                                      orig_to_tok_map=orig_to_tok_map,
                                                      tokenizer=self.tokenizer)

            # Actual prediction happens here
            inputs = {
                'input_ids': input_ids.to(self.device),
                'token_type_ids': torch.zeros_like(input_ids).to(self.device),
                'attention_mask': attention_mask.to(self.device),
            }

            outputs = self.model(**inputs)
            logits = outputs[0]

            y_preds = torch.argmax(logits, dim=-1)

            # If y_preds 2D matrix, this is a STM. Add dummy first dimension
            if len(y_preds.size()) < 3:
                y_preds = torch.unsqueeze(y_preds, dim=0)

            # TODO (John): There has got to be a better way to do this?
            # Mask [PAD] and WORDPIECE tokens
            y_preds_masked = []
            for output, dataset in zip(y_preds, self.datasets):
                y_preds_masked.append([])
                for y_pred, tok_map in zip(output, orig_to_tok_map):
                    orig_token_indices = torch.as_tensor(
                        [i for i in tok_map if i != TOK_MAP_PAD],
                        device=self.device,
                        dtype=torch.long
                    )

                    masked_logits = torch.index_select(y_pred, -1, orig_token_indices)
                    masked_logits = masked_logits.detach().tolist()

                    predicted_labels = [dataset.idx_to_tag['ent'][idx] for idx in masked_logits]

                    y_preds_masked[-1].append(predicted_labels)

            # If STM, return only a list of lists
            if len(y_preds_masked) == 1:
                y_preds_masked = y_preds_masked[0]

        return y_preds_masked

    def prepare_optimizers(self):
        """Returns a list of PyTorch optimizers, one per output layer in `self.model`.

        For each output layer in `self.model`, creates an optmizer based on the given config at
        `self.config`. For a single-task model, the returned list will be of length 1. For
        a multi-task model, the returned list will be of length `len(self.model.ouput)`.

        Returns:
            A list of PyTorch optimizers initiated from the given config at `self.config`.
        """
        optimizers = [bert_utils.get_bert_optimizer(self.model, self.config)
                      for _, _ in enumerate(self.num_labels)]

        return optimizers
