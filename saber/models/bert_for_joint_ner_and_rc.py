import logging
from itertools import zip_longest
from pprint import pprint

import torch
from pytorch_pretrained_bert import BertTokenizer
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from .. import constants
from ..constants import TOK_MAP_PAD
from ..constants import WORDPIECE
from ..utils import bert_utils
from ..utils import data_utils
from ..utils import model_utils
from .base_model import BasePyTorchModel
from .modules.bert_for_joint_entity_and_relation_classification import \
    BertForJointEntityAndRelationClassification

LOGGER = logging.getLogger(__name__)

# TODO (johngiorgi): Test that saving loading from CPU/GPU works as expected


class BertForJointNERAndRC(BasePyTorchModel):
    """A PyTorch implementation of a BERT model for joint NER and RC.

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
        super(BertForJointNERAndRC, self).__init__(config, datasets, **kwargs)

        # Place the model on the CPU by default
        self.device = torch.device("cpu")
        self.n_gpus = 0

        self.num_ent_labels, self.num_rel_labels = [], []
        for dataset in self.datasets:
            if WORDPIECE not in dataset.type_to_idx['ent']:
                dataset.type_to_idx['ent'][WORDPIECE] = len(dataset.type_to_idx['ent'])
                dataset.get_idx_to_tag()
            self.num_ent_labels.append(len(dataset.idx_to_tag['ent']))
            self.num_rel_labels.append(len(dataset.idx_to_tag['rel']))

        # Name or path of a pre-trained BERT model
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.model_name = 'bert-ner-rc'

    def load(self, model_filepath):
        """Loads a PyTorch BERT model for named entity recognition from disk.

        Loads a PyTorch `BertForJointEntityAndRelationClassification` model and its corresponding
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

        model = BertForJointEntityAndRelationClassification.from_pretrained(
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
        """Specifies a PyTorch BERT model for named entity recognition and its tokenizer.

        Returns:
            The loaded PyTorch BERT model and the corresponding Tokenizer.
        """
        # If this is one of our preprocessed BERT models, download it from GDrive first
        if self.pretrained_model_name_or_path in constants.PRETRAINED_MODELS:
            self.pretrained_model_name_or_path = \
                model_utils.download_model_from_gdrive(self.pretrained_model_name_or_path)

        # TODO (John): Temporary, need a better API for this.
        rel_class_weight = self.datasets[0].compute_class_weight()['rel'].tolist()
        rel_class_weight = torch.tensor([1] + rel_class_weight)

        model = BertForJointEntityAndRelationClassification.from_pretrained(
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
        """Returns a list containing the training data for each dataset at `self.datasets`.

        For each dataset at `self.datasets`, collects the data to be used for training.
        Each dataset is represented by a dictionary, where the keys 'x_<partition>' and
        'y_<partition>' contain the inputs and targets for each partition 'train', 'valid', and
        """
        training_data = []
        for model_idx, dataset in enumerate(self.datasets):
            processed_dataset = bert_utils.process_dataset_for_bert(dataset, self.tokenizer)
            dataloaders = bert_utils.get_dataloader_for_bert(processed_dataset=processed_dataset,
                                                             batch_size=self.config.batch_size,
                                                             model_idx=model_idx)

            processed_dataset.update(dataloaders)
            training_data.append(processed_dataset)

        return training_data

    # TODO (John): Way too much duplicated code (DRY). Fix!
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
            # if training_data[0]['x_valid'] is None:
            #   training_data = data_utils.collect_valid_data(training_data)

            # get list of Keras Callback objects for computing/storing metrics
            metrics = model_utils.setup_metrics_callback(model=self,
                                                         config=self.config,
                                                         datasets=self.datasets,
                                                         training_data=training_data,
                                                         output_dir=output_dir,)

            # Collect dataloaders, we don't need the other data
            dataloaders = [data['dataloader_train'] for data in training_data]
            total = len(max(dataloaders, key=len))

            for epoch in range(self.config.epochs):

                self.model.train()

                train_loss_ner = [0] * len(self.num_ent_labels)
                train_loss_rc = [0] * len(self.num_ent_labels)
                train_loss_joint = [0] * len(self.num_ent_labels)
                train_steps = [0] * len(self.num_ent_labels)

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
                            (batch_indices, input_ids, attention_mask, ent_labels, orig_to_tok_map,
                             model_idx) = batch

                            input_ids = input_ids.to(self.device)
                            attention_mask = attention_mask.to(self.device)
                            ent_labels = ent_labels.to(self.device)
                            model_idx = model_idx[0].item()

                            rel_labels = [training_data[model_idx]['rel_labels_train'][idx]
                                          for idx in batch_indices]

                            optimizer = optimizers[model_idx]
                            optimizer.zero_grad()

                            _, _, _, loss_ner, loss_re = self.model(
                                input_ids=input_ids,
                                orig_to_tok_map=orig_to_tok_map,
                                token_type_ids=torch.zeros_like(input_ids),
                                attention_mask=attention_mask,
                                ent_labels=ent_labels,
                                rel_labels=rel_labels
                            )
                            if epoch > 0:
                                loss = loss_ner + loss_re
                            else:
                                loss = loss_ner

                            loss.backward()

                            # Loss is a vector of size n_gpus, need to average if more than 1
                            if self.n_gpus > 1:
                                loss = loss.mean()

                            # Track train loss
                            train_loss_ner[model_idx] += loss_ner.item()
                            train_loss_rc[model_idx] += loss_re.item()
                            train_loss_joint[model_idx] += loss.item()
                            train_steps[model_idx] += 1

                            optimizer.step()

                            # Update train loss in progress bar
                            postfix = {
                                'NER_loss': f'{train_loss_ner[model_idx] / train_steps[model_idx]:.4f}',
                                'RC_loss': f'{train_loss_rc[model_idx] / train_steps[model_idx]:.4f}',
                                'joint_loss': f'{train_loss_joint[model_idx] / train_steps[model_idx]:.4f}',
                            }
                            pbar.set_postfix(postfix)

                for metric in metrics:
                    # Need to feed epoch argument manually, as this is a keras callback object
                    metric.on_epoch_end(epoch=epoch)

                pbar.close()

        def cross_validation(training_data, output_dir, optimizers):
            # Get the train / valid partitioned data for all datasets and all folds
            training_data_cv = data_utils.collect_cv_data(training_data, self.config.k_folds)
            training_data = []
            for model_idx, _ in enumerate(training_data_cv):
                training_data.append([])
                for fold, data in enumerate(training_data_cv[model_idx]):
                    data['x_train'][0] = torch.as_tensor(data['x_train'][0])
                    data['x_train'][-1] = torch.as_tensor(data['x_train'][-1])
                    data['y_train'] = torch.as_tensor(data['y_train'])
                    data['orig_to_tok_map_train'] = torch.as_tensor(data['orig_to_tok_map_train'])

                    training_data[-1].append(data)

                    dataloaders = \
                        bert_utils.get_dataloader_for_bert(processed_dataset=data,
                                                           batch_size=self.config.batch_size,
                                                           model_idx=model_idx)
                    data.update(dataloaders)

            # Training loop
            for fold in range(self.config.k_folds):

                # Get list of Keras Callback objects for computing/storing metrics
                metrics = model_utils.setup_metrics_callback(model=self,
                                                             config=self.config,
                                                             datasets=self.datasets,
                                                             training_data=training_data,
                                                             output_dir=output_dir,
                                                             fold=fold)

                # Collect dataloaders, we don't need the other data
                dataloaders = [data[fold]['dataloader_train'] for data in training_data]
                total = len(max(dataloaders, key=len))

                for epoch in range(self.config.epochs):

                    self.model.train()

                    train_loss_ner = [0] * len(self.num_ent_labels)
                    train_loss_rc = [0] * len(self.num_ent_labels)
                    train_loss_joint = [0] * len(self.num_ent_labels)
                    train_steps = [0] * len(self.num_ent_labels)

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
                                (batch_indices, input_ids, attention_mask, ent_labels,
                                 orig_to_tok_map, model_idx) = batch

                                input_ids = input_ids.to(self.device)
                                attention_mask = attention_mask.to(self.device)
                                ent_labels = ent_labels.to(self.device)
                                model_idx = model_idx[0].item()

                                rel_labels = \
                                    [training_data[model_idx][fold]['rel_labels_train'][idx]
                                     for idx in batch_indices]

                                optimizer = optimizers[model_idx]
                                optimizer.zero_grad()

                                _, _, _, loss_ner, loss_re = self.model(
                                    input_ids=input_ids,
                                    orig_to_tok_map=orig_to_tok_map,
                                    token_type_ids=torch.zeros_like(input_ids),
                                    attention_mask=attention_mask,
                                    ent_labels=ent_labels,
                                    rel_labels=rel_labels,
                                )
                                if epoch > 0:
                                    loss = loss_ner + loss_re
                                else:
                                    loss = loss_ner

                                loss.backward()

                                # Loss is a vector of size n_gpus, need to average if more than 1
                                if self.n_gpus > 1:
                                    loss = loss.mean()

                                # Track train loss
                                train_loss_ner[model_idx] += loss_ner
                                train_loss_rc[model_idx] += loss_re.item()
                                train_loss_joint[model_idx] += loss.item()
                                train_steps[model_idx] += 1

                                optimizer.step()

                            # Update train loss in progress bar
                            postfix = {
                                'NER_loss': f'{train_loss_ner[model_idx] / train_steps[model_idx]:.4f}',
                                'RC_loss': f'{train_loss_rc[model_idx] / train_steps[model_idx]:.4f}',
                                'joint_loss': f'{train_loss_joint[model_idx] / train_steps[model_idx]:.4f}',
                            }
                            pbar.set_postfix(postfix)

                    for metric in metrics:
                        # Need to feed epoch argument manually, as this is a keras callback object
                        metric.on_epoch_end(epoch=epoch)

                    pbar.close()

                # Clear and rebuild the model at end of each fold (except for the last fold)
                if fold < self.config.k_folds - 1:
                    self.reset_model()
                    torch.cuda.empty_cache()
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

        y_true, y_pred, y_true_rel, y_pred_rel = [], [], [], []
        eval_loss, eval_steps = 0, 0

        dataloader = training_data[f'dataloader_{partition}']

        for batch in dataloader:
            (batch_indices, input_ids, attention_mask, ent_labels, orig_to_tok_map,
             model_idx) = batch

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            ent_labels = ent_labels.to(self.device)
            model_idx = model_idx[0].item()

            rel_labels = [training_data[f'rel_labels_{partition}'][idx] for idx in batch_indices]

            with torch.no_grad():
                ner_logits, rc_logits, rc_labels, ner_loss, rc_loss = self.model(
                    input_ids=input_ids,
                    orig_to_tok_map=orig_to_tok_map,
                    token_type_ids=torch.zeros_like(input_ids),
                    attention_mask=attention_mask,
                    ent_labels=ent_labels,
                    rel_labels=rel_labels
                )

                loss = ner_loss + rc_loss
                ner_logits = ner_logits.argmax(dim=-1)
                rc_logits = rc_logits.argmax(dim=-1)

                y_true_rel.extend(rc_labels.view(-1).tolist())
                y_pred_rel.extend(rc_logits.view(-1).tolist())

                # Loss object is a vector of size n_gpus, need to average if more than 1
                if self.n_gpus > 1:
                    loss = loss.mean()

            # Mask [PAD] and WORDPIECE tokens
            for logits_, labels_, tok_map in zip(ner_logits, ent_labels, orig_to_tok_map):
                orig_token_indices = torch.as_tensor([i for i in tok_map if i != TOK_MAP_PAD])
                orig_token_indices = orig_token_indices.to(self.device).long()

                masked_logits = torch.index_select(logits_, -1, orig_token_indices)
                masked_logits = masked_logits.tolist()

                masked_labels = torch.index_select(labels_, -1, orig_token_indices)
                masked_labels = masked_labels.tolist()

                y_true.extend(masked_labels)
                y_pred.extend(masked_logits)

            eval_loss += loss.item()
            eval_steps += 1

        # TODO (John): This should be moved to metrics

        labels = list(set(self.datasets[model_idx].idx_to_tag['rel'].keys()))
        labels.sort()

        print(y_true_rel[:20])
        print(y_pred_rel[:20])

        prfs = precision_recall_fscore_support(y_true_rel, y_pred_rel, labels=labels)
        prfs_macro = precision_recall_fscore_support(y_true_rel, y_pred_rel,
                                                     labels=labels[1:],
                                                     average='macro')

        pprint(prfs)

        print(f'Macro F1 Score: {prfs_macro[2]:.2%}')

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
            If `self.models` has a single output layer (TODO):
                A list of lists, containing the predicted labels for `tokens`.
            If `self.models` has multiple output layers (TODO):
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
            ner_logits, rc_logits = self.model(
                input_ids=input_ids,
                orig_to_tok_map=orig_to_tok_map,
                token_type_ids=torch.zeros_like(input_ids),
                attention_mask=attention_masks)

        ner_preds = ner_logits.argmax(dim=-1)
        rc_preds = rc_logits.argmax(dim=-1) if rc_logits is not None else rc_logits

        # If y_preds 2D matrix, this is a STM. Add dummy first dimension
        if len(list(ner_preds.shape)) < 3:
            ner_preds = torch.unsqueeze(ner_preds, dim=0)

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

        return ner_preds_masked, rc_preds

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
