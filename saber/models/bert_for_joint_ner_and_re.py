from itertools import zip_longest

import torch
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm

from .. import constants
from ..constants import TOK_MAP_PAD
from ..constants import WORDPIECE
from ..utils import bert_utils
from ..utils import data_utils
from ..utils import model_utils
from .base_model import BaseModel
from .modules.bert_for_joint_entity_and_relation_classification import \
    BertForJointEntityAndRelationExtraction


class BertForJointNERAndRE(BaseModel):
    """A PyTorch implementation of a BERT model for joint named entity recognition (NER) and
    relation extraction (RE).

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
        - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding:
            https://arxiv.org/abs/1810.04805
        - PyTorch Implementation of BERT by Huggingface:
            https://github.com/huggingface/pytorch-pretrained-BERT
    """
    def __init__(self, config, datasets, pretrained_model_name_or_path='bert-base-cased', **kwargs):
        super(BertForJointNERAndRE, self).__init__(config, datasets, **kwargs)

        # Place the model on the CPU by default
        self.device = torch.device("cpu")
        self.n_gpus = 0

        self.num_ent_labels, self.num_rel_labels = [], []
        for dataset in self.datasets:
            # Required for dataset to be compatible with BERTs wordpiece tokenization
            if WORDPIECE not in dataset.type_to_idx['ent']:
                dataset.type_to_idx['ent'][WORDPIECE] = len(dataset.type_to_idx['ent'])
                dataset.get_idx_to_tag()
            self.num_ent_labels.append(len(dataset.idx_to_tag['ent']))
            self.num_rel_labels.append(len(dataset.idx_to_tag['rel']))

        # Name or path of a pre-trained BERT model
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.model_name = 'bert-ner-rc'

    def load(self, model_filepath):
        """Loads a PyTorch `BertForJointEntityAndRelationExtraction` model from disk.

        Loads a PyTorch `BertForJointEntityAndRelationExtraction` model and its corresponding
        tokenizer (both saved with `BertForNER.save()`) from disk by loading its architecture and
        weights from a `.bin` file at `model_filepath`.

        Args:
            model_filepath (str): filepath to the models architecture (`.bin` file).

        Returns:
            The loaded PyTorch BERT model and the corresponding Tokenizer.
        """
        model_state_dict = torch.load(model_filepath, map_location=lambda storage, loc: storage)

        # A trick to get the number of labels for each classifier
        # TODO (John): These are lists because in the future we would like to support MTL.
        num_ent_labels = [model_state_dict['ent_classifier.weight'].size(0)]
        num_rel_labels = [model_state_dict['rel_classifier.linear.weight'].size(0)]

        model = BertForJointEntityAndRelationExtraction.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            idx_to_ent=[dataset.idx_to_tag['ent'] for dataset in self.datasets],
            num_ent_labels=num_ent_labels,
            num_rel_labels=num_rel_labels,
        )
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name_or_path,
                                                  do_lower_case=False)

        # Get the device the model will live on, along with number of GPUs available
        self.device, self.n_gpus = model_utils.get_device(model)

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def specify(self):
        """Specifies a PyTorch `BertForJointEntityAndRelationExtraction` model and its tokenizer.

        Returns:
            The loaded PyTorch `BertForJointEntityAndRelationExtraction` model and its
            corresponding Tokenizer.
        """
        # If this is one of our preprocessed BERT models, download it from GDrive first
        if self.pretrained_model_name_or_path in constants.PRETRAINED_MODELS:
            self.pretrained_model_name_or_path = \
                model_utils.download_model_from_gdrive(self.pretrained_model_name_or_path)

        # TODO (John): Temporary, need a better API for this.
        rel_class_weight = self.datasets[0].compute_class_weight()['rel'].tolist()
        rel_class_weight = torch.tensor([1] + rel_class_weight)

        model = BertForJointEntityAndRelationExtraction.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            idx_to_ent=[dataset.idx_to_tag['ent'] for dataset in self.datasets],
            num_ent_labels=self.num_ent_labels,
            num_rel_labels=self.num_rel_labels,
            rel_class_weight=rel_class_weight,
        )
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name_or_path,
                                                  do_lower_case=False)

        # Get the device the model will live on, along with number of GPUs available
        self.device, self.n_gpus = model_utils.get_device(model)

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def prepare_data_for_training(self):
        """Returns a list containing data which has been processed for training with `self.model`.

        For  each dataset at `self.datasets`, processes the data to be used for training and/or
        evaluation with `self.model`. Returns a list of dictionaries, of length `len(self.datasets)`
        containing a PyTorch Dataloader for each of a datasets partitions (e.g.
        `training_data[partition][0]['train']['dataloader']`) contains the dataloader for the
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
            processed_dataset = data_utils.get_validation_set(self.config, processed_dataset)

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

        metrics = model_utils.setup_metrics_callback(config=self.config,
                                                     model=self,
                                                     datasets=self.datasets,
                                                     training_data=training_data,
                                                     output_dirs=output_dirs,)

        # Training loop
        k_folds = len(training_data[0])
        for fold in range(k_folds):
            # Collect dataloaders, we don't need the other data
            dataloaders = [data[fold]['train']['dataloader'] for data in training_data]
            total = len(max(dataloaders, key=len))

            for epoch in range(self.config.epochs):

                self.model.train()

                train_ner_loss = [0] * len(self.num_ent_labels)
                train_re_loss = [0] * len(self.num_ent_labels)
                train_loss_joint = [0] * len(self.num_ent_labels)
                train_steps = [0] * len(self.num_ent_labels)

                # Setup a progress bar
                if k_folds > 1:
                    fold_and_epoch = (fold + 1, k_folds, epoch + 1, self.config.epochs)
                    desc = 'Fold: {}/{}, Epoch: {}/{}'.format(*fold_and_epoch)
                else:
                    desc = f'Epoch: {epoch + 1}/{self.config.epochs}'

                pbar = tqdm(zip_longest(*dataloaders),
                            unit='batch',
                            desc=desc,
                            total=total,
                            dynamic_ncols=True)

                for _, batches in enumerate(pbar):
                    for batch in batches:
                        if batch is not None:
                            (batch_indices, input_ids, attention_mask, ent_labels,
                                orig_to_tok_map, model_idx) = batch

                            input_ids = input_ids.to(self.device)
                            attention_mask = attention_mask.to(self.device)
                            ent_labels = ent_labels.to(self.device)
                            model_idx = model_idx[0].item()

                            rel_labels = \
                                [training_data[model_idx][fold]['train']['rel_labels'][idx]
                                    for idx in batch_indices]

                            optimizer = optimizers[model_idx]
                            optimizer.zero_grad()

                            _, _, ner_loss, re_loss, _ = self.model(
                                input_ids=input_ids,
                                orig_to_tok_map=orig_to_tok_map,
                                token_type_ids=torch.zeros_like(input_ids),
                                attention_mask=attention_mask,
                                ent_labels=ent_labels,
                                rel_labels=rel_labels,
                            )

                            # TODO (John): Hotfix, this should be added to a config
                            if epoch > 0:
                                loss = ner_loss + re_loss
                            else:
                                loss = ner_loss

                            loss.backward()

                            # Loss is a vector of size n_gpus, need to average if more than 1
                            if self.n_gpus > 1:
                                loss = loss.mean()

                            # Track train loss
                            train_ner_loss[model_idx] += ner_loss
                            train_re_loss[model_idx] += re_loss.item()
                            train_loss_joint[model_idx] += loss.item()
                            train_steps[model_idx] += 1

                            optimizer.step()

                        # Update train loss in progress bar
                        postfix = {
                            'NER_loss': f'{train_ner_loss[model_idx] / train_steps[model_idx]:.4f}',
                            'RE_loss': f'{train_re_loss[model_idx] / train_steps[model_idx]:.4f}',
                            'joint_loss': f'{train_loss_joint[model_idx] / train_steps[model_idx]:.4f}',
                        }
                        pbar.set_postfix(postfix)

                for metric in metrics:
                    # Need to feed epoch argument manually, as this is a Keras callback object
                    metric.on_epoch_end()

                pbar.close()

            # Clear and rebuild the model at end of each fold (except for the last fold)
            if k_folds > 1 and fold < k_folds - 1:
                self.reset_model()
                optimizers = self.prepare_optimizers()

                for metric in metrics:
                    metric.on_fold_end()  # bumps internal fold counter

        return metrics

    def evaluate(self, training_data, partition='train', model_idx=None):
        """Perform a prediction step using `self.model` on `training_data[partition]`.

        Performs prediction for the model at `self.model` and returns a four-tuple containing the
        true (gold) NER labels and corresponding predicted labels, and the true (gold) RE labels and
        their corresponding predicted labels.

        Args:
            training_data (dict): Contains the data (at key `partition_dataloader`).
            partition (str): Which partition to perform a prediction step on, must be one of
                'train', 'valid', or 'test'.
            model_idx (None): Unused argument. Retained for compatibility.

        Returns:
            A four-tuple containing the gold NER label sequences and corresponding predicted labels
            and the gold RE label sequences and corresponding predicted labels.
        """
        self.model.eval()

        y_true_ner, y_pred_ner, y_true_re, y_pred_re = [], [], [], []
        eval_ner_loss, eval_re_loss, eval_joint_loss, eval_steps = 0, 0, 0, 0

        dataloader = training_data[partition]['dataloader']

        for batch in dataloader:
            (batch_indices, input_ids, attention_mask, ent_labels, orig_to_tok_map,
             model_idx) = batch

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            ent_labels = ent_labels.to(self.device)
            model_idx = model_idx[0].item()

            rel_labels = [training_data[partition]['rel_labels'][idx] for idx in batch_indices]

            with torch.no_grad():
                ner_logits, re_logits, ner_loss, re_loss, re_labels = self.model(
                    input_ids=input_ids,
                    orig_to_tok_map=orig_to_tok_map,
                    token_type_ids=torch.zeros_like(input_ids),
                    attention_mask=attention_mask,
                    ent_labels=ent_labels,
                    rel_labels=rel_labels
                )

                loss = ner_loss + re_loss
                ner_preds = ner_logits.argmax(dim=-1)
                re_preds = re_logits.argmax(dim=-1)

                # Loss object is a vector of size n_gpus, need to average if more than 1
                if self.n_gpus > 1:
                    loss = loss.mean()

                # Mask [PAD] and WORDPIECE tokens
                for preds, labels, tok_map in zip(ner_preds, ent_labels, orig_to_tok_map):
                    orig_token_indices = torch.as_tensor([i for i in tok_map if i != TOK_MAP_PAD],
                                                         device=self.device,
                                                         dtype=torch.long)

                    masked_labels = torch.index_select(labels, -1, orig_token_indices).tolist()
                    masked_preds = torch.index_select(preds, -1, orig_token_indices).tolist()

                    # Map entity predictions to tags
                    y_true_ner.append([self.datasets[model_idx].idx_to_tag['ent'][idx]
                                       for idx in masked_labels])
                    y_pred_ner.append([self.datasets[model_idx].idx_to_tag['ent'][idx]
                                       for idx in masked_preds])

                # Map relation predictions to tags
                y_true_re.append([self.datasets[model_idx].idx_to_tag['rel'][idx]
                                  for idx in re_labels.view(-1).tolist()])
                y_pred_re.append([self.datasets[model_idx].idx_to_tag['rel'][idx]
                                  for idx in re_preds.view(-1).tolist()])

                # TODO (John): We don't actually do anything with this?
                eval_ner_loss += ner_loss.item()
                eval_re_loss += re_loss.item()
                eval_joint_loss += loss.item()
                eval_steps += 1

        return y_true_ner, y_pred_ner, y_true_re, y_pred_re

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

        # Prepare data for input to model
        bert_tokens, orig_to_tok_map = bert_utils.wordpiece_tokenize_sents(tokens, self.tokenizer)
        input_ids, orig_to_tok_map, attention_masks = \
            bert_utils.index_pad_mask_bert_tokens(tokens=bert_tokens,
                                                  orig_to_tok_map=orig_to_tok_map,
                                                  tokenizer=self.tokenizer)

        input_ids = input_ids.to(self.device)
        orig_to_tok_map = orig_to_tok_map.to(self.device)
        attention_masks = attention_masks.to(self.device)

        # Actual prediction happens here
        with torch.no_grad():
            ner_logits, re_logits = self.model(
                input_ids=input_ids,
                orig_to_tok_map=orig_to_tok_map,
                token_type_ids=torch.zeros_like(input_ids),
                attention_mask=attention_masks)

        ner_preds = ner_logits.argmax(dim=-1)
        re_preds = re_logits.argmax(dim=-1) if re_logits is not None else re_logits

        # If y_preds 2D matrix, this is a STM. Add dummy first dimension
        if len(list(ner_preds.size())) < 3:
            ner_preds = torch.unsqueeze(ner_preds, dim=0)

        # TODO (John): There has got to be a better way to do this?
        # Mask [PAD] and WORDPIECE tokens
        ner_preds_masked = []
        for output, dataset in zip(ner_preds, self.datasets):
            ner_preds_masked.append([])
            for pred, tok_map in zip(output, orig_to_tok_map):
                orig_token_indices = torch.as_tensor([i for i in tok_map if i != TOK_MAP_PAD])
                orig_token_indices = orig_token_indices.to(self.device).long()

                masked_logits = torch.index_select(pred, -1, orig_token_indices)
                masked_logits = masked_logits.detach().tolist()

                predicted_labels = [dataset.idx_to_tag['ent'][idx] for idx in masked_logits]
                ner_preds_masked[-1].append(predicted_labels)

        # If STM, return only a list of lists
        if len(ner_preds_masked) == 1:
            ner_preds_masked = ner_preds_masked[0]

        return ner_preds_masked, re_preds

    def prepare_optimizers(self):
        """Returns a list of PyTorch optimizers, one per output layer in `self.model`.

        For each output layer in `self.model`, creates an optmizer based on the given config at
        `self.config`. For a single-task model, the returned list will be of length 1. For
        a multi-task model, the returned list will be of length `len(self.model.ouput)`.

        Returns:
            A list of PyTorch optimizers initiated from the given config at `self.config`.
        """
        optimizers = [bert_utils.get_bert_optimizer(self.model, self.config)
                      for _, _ in enumerate(self.num_ent_labels)]

        return optimizers
