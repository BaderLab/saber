"""Contains the Multi-task BiLSTM-CRF (bilstm-crf-ner) Keras model for sequence labelling.
"""
import logging
import random

import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers import TimeDistributed
from keras.models import Input
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras_contrib.layers.crf import CRF
from keras_contrib.losses.crf_losses import crf_loss

from ..constants import PAD
from ..constants import PAD_VALUE
from ..constants import UNITS_CHAR_LSTM
from ..constants import UNITS_WORD_LSTM
from ..preprocessor import Preprocessor
from ..utils import data_utils
from ..utils import model_utils
from .base_model import BaseKerasModel

LOGGER = logging.getLogger(__name__)


class BiLSTMCRF(BaseKerasModel):
    """A Keras implementation of a multi-task BiLSTM-CRF model for named entity recognition (NER).

    A BiLSTM-CRF for NER implementation in Keras. Supports multi-task learning by default, just pass
    multiple Dataset objects via `datasets` to the constructor and the model will share the
    parameters of all layers, except for the final output layer, across all datasets, where each
    dataset represents a NER task.

    Args:
        config (Config): Contains a set of harmonzied arguments provided in a *.ini file and,
            optionally, from the command line.
        datasets (list): A list of Dataset objects.
        embeddings (Embeddings): Optional, an object containing loaded word embeddings. If None,
            embeddings are randomly initialized and updated during training. Defaults to None.

    References:
        - https://www.biorxiv.org/content/10.1101/526244v1
        - https://arxiv.org/pdf/1512.05287.pdf
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        super().__init__(config, datasets, embeddings, **kwargs)

        self.model_name = 'bilstm-crf-ner'

    def load(self, model_filepath, weights_filepath):
        """Load a BiLSTM-CRF model from disk.

        Loads a BiLSTM-CRF Keras model from disk by loading its architecture from a `.json` file at
        `model_filepath` and its weights from a `.hdf5` file at `model_filepath`.

        Args:
            model_filepath (str): Filepath to the models architecture (`.json` file).
            weights_filepath (str): Filepath to the models weights (`.hdf5` file).

        Returns:
            The BiLSTM-CRF Keras `Model` object that was saved to disk.
        """
        model = super().load(model_filepath, weights_filepath, custom_objects={'CRF': CRF})

        return model

    def specify(self):
        """Specifies a BiLSTM-CRF for named entity recognition (NER) using Keras.

        Implements a hybrid bidirectional long short-term memory network-conditional random
        field (BiLSTM-CRF) multi-task model for sequence tagging. If `len(self.datasets) > 1`, a
        single input, multi-output model is created, with one CRF output layer per dataset in
        `self.datasets`.

        Returns:
            The Keras `Model` object that was initialized.
        """
        # Word-level embedding layer
        if self.embeddings is None:
            # +1 to account for pad
            word_embeddings = Embedding(input_dim=len(self.datasets[0].type_to_idx['word']) + 1,
                                        output_dim=self.config.word_embed_dim,
                                        mask_zero=True,
                                        name="word_embedding_layer")
        else:
            word_embeddings = Embedding(input_dim=self.embeddings.num_embed,
                                        output_dim=self.embeddings.dimension,
                                        mask_zero=True,
                                        weights=[self.embeddings.matrix],
                                        trainable=self.config.fine_tune_word_embeddings,
                                        name="word_embedding_layer")

        # Character-level embedding layer
        char_embeddings = Embedding(input_dim=len(self.datasets[0].type_to_idx['char']) + 1,
                                    output_dim=self.config.char_embed_dim,
                                    mask_zero=True,
                                    name="char_embedding_layer")

        # Word-level input embeddings
        word_ids = Input(shape=(None, ), dtype='int32', name='word_id_inputs')
        word_embeddings = word_embeddings(word_ids)

        # Character-level input embeddings
        char_ids = Input(shape=(None, None), dtype='int32', name='char_id_inputs')
        char_embeddings = char_embeddings(char_ids)

        # Char-level BiLSTM
        char_BiLSTM = TimeDistributed(Bidirectional(LSTM(UNITS_CHAR_LSTM // 2)),
                                      name='character_BiLSTM')
        # Word-level BiLSTM
        word_BiLSTM_1 = Bidirectional(LSTM(units=UNITS_WORD_LSTM // 2,
                                           return_sequences=True,
                                           dropout=self.config.dropout_rate['input'],
                                           recurrent_dropout=self.config.dropout_rate['recurrent']),
                                      name="word_BiLSTM_1")
        word_BiLSTM_2 = Bidirectional(LSTM(units=UNITS_WORD_LSTM // 2,
                                           return_sequences=True,
                                           dropout=self.config.dropout_rate['input'],
                                           recurrent_dropout=self.config.dropout_rate['recurrent']),
                                      name="word_BiLSTM_2")

        # Character-level BiLSTM + dropout. Spatial dropout applies the same dropout mask to all
        # timesteps which is necessary to implement variational dropout
        # (https://arxiv.org/pdf/1512.05287.pdf)
        char_BiLSTM = char_BiLSTM(char_embeddings)
        if self.config.variational_dropout:
            LOGGER.info('Used variational dropout')
            char_BiLSTM = SpatialDropout1D(self.config.dropout_rate['output'])(char_BiLSTM)

        # Concatenate word- and char-level embeddings + dropout
        char_enhanced_word_embeddings = Concatenate()([word_embeddings, char_BiLSTM])
        char_enhanced_word_embeddings = \
            Dropout(self.config.dropout_rate['output'])(char_enhanced_word_embeddings)

        # Char-enhanced-level BiLSTM + dropout
        char_enhanced_word_embeddings = word_BiLSTM_1(char_enhanced_word_embeddings)
        if self.config.variational_dropout:
            char_enhanced_word_embeddings = \
                SpatialDropout1D(self.config.dropout_rate['output'])(char_enhanced_word_embeddings)
        char_enhanced_word_embeddings = word_BiLSTM_2(char_enhanced_word_embeddings)
        if self.config.variational_dropout:
            char_enhanced_word_embeddings = \
                SpatialDropout1D(self.config.dropout_rate['output'])(char_enhanced_word_embeddings)

        # Get all unique tag types across all datasets
        all_tags = [ds.type_to_idx['ent'] for ds in self.datasets]
        all_tags = set(x for l in all_tags for x in l)

        # Feedforward before CRF, maps each time step to a vector
        dense_layer = TimeDistributed(Dense(len(all_tags), activation=self.config.activation),
                                      name='dense_layer')
        label_predictions = dense_layer(char_enhanced_word_embeddings)

        # CRF output layer(s)
        output_layers = []
        for i, dataset in enumerate(self.datasets):
            crf = CRF(len(dataset.type_to_idx['ent']), name=f'crf_{i}')
            output_layers.append(crf(label_predictions))

        # Fully specified model
        # https://github.com/keras-team/keras/blob/bf1378f39d02b7d0b53ece5458f9275ac8208046/keras/utils/multi_gpu_utils.py
        with tf.device('/cpu:0'):
            model = Model(inputs=[word_ids, char_ids], outputs=output_layers)

        self.model = model

        return model

    # TODO (John): This should relly be in the BaseKerasModel class
    def compile(self, loss=crf_loss, optimizer=None, loss_weights=None):
        """Compiles the Keras model `self.model`.

        Compiles the Keras model `self.model`. If multiple GPUs are detected, a model capable of
        training on all of them is compiled.

        Args:
            loss: String (name of objective function) or objective function. See Keras losses. If
                the model has multiple outputs, you can use a different loss on each output by
                passing a dictionary or a list of losses. The loss value that will be minimized by
                the model will then be the sum of all individual losses. If a single loss is passed
                for a multi-output model, this loss function will be duplicated, one per output
                layer.
            optimizer (keras.Optimizer) Optional, Keras optimizer instance. If None, the optimizer
                with name `self.config.optimizer` is instantiated.
            loss_weights (list or dict): Optional, list or dictionary specifying scalar coefficients
                (Python floats) to weight the loss contributions of different model outputs. The
                loss value that will be minimized by the model will then be the weighted sum of all
                individual losses, weighted by the `loss_weights` coefficients. If a list, it is
                expected to have a 1:1 mapping to the model's outputs. If a tensor, it is expected
                to map output names (strings) to scalar coefficients.

        Returns:
            `self.model` after calling `self.model.compile()`.
        """
        # parallize the model if multiple GPUs are available
        # https://github.com/keras-team/keras/pull/9226#issuecomment-361692460
        # awfully bad practice but this was the example given by Keras documentation
        try:
            self.model = multi_gpu_model(self.model)
            LOGGER.info('Compiling the model on multiple GPUs.')
        except:
            LOGGER.info('Compiling the model on a single CPU or GPU.')

        if optimizer is None:
            optimizer = model_utils.get_keras_optimizer(optimizer=self.config.optimizer,
                                                        lr=self.config.learning_rate,
                                                        decay=self.config.decay,
                                                        clipnorm=self.config.grad_norm)

        # Create one loss function per output layer if only a single loss function was provided
        if isinstance(self.model.output, list) and not isinstance(loss, list):
            loss = [loss for _, _ in enumerate(self.model.output)]

        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

        return self.model

    def prepare_data_for_training(self):
        """Returns a list containing data which has been processed for training with `self.model`.

        For each dataset at `self.datasets`, processes the data to be used for training and/or
        evaluation with `self.model`. Returns a list of dictionaries, of length `len(self.datasets)`
        keyed by partition, where each dictionary contains the inputs at `[partition]['x']` and the
        targets at `[partition]['y']` for `partition` of the corresponding dataset.

        Returns:
            A list of dictionaries, of length `len(self.datasets)`, containing the training data for
            each dataset at `self.datasets`.
        """
        training_data = []
        for dataset in self.datasets:
            training_data.append({})
            for partition, filepath in dataset.dataset_folder.items():
                if filepath is not None:
                    training_data[-1][partition] = {
                        'x': [dataset.idx_seq[partition]['word'],
                              dataset.idx_seq[partition]['char']],
                        # One-hot encode our targets
                        'y': to_categorical(dataset.idx_seq[partition]['ent'])
                    }
                else:
                    training_data[-1][partition] = None

            training_data[-1] = data_utils.get_validation_set(self.config, training_data[-1])

            # A hack to ensure that training data is always a list (datasets) of lists (folds)
            if not isinstance(training_data[-1], list):
                training_data[-1] = [training_data[-1]]

        return training_data

    def train(self):
        """Co-ordinates the training of the Keras model `self.model`.

        Co-ordinates the training of the Keras model `self.model`. Minimally expects a train
        partition to be supplied in the Dataset objects at `self.datasets`.
        """
        # Gather everything we need to run a training session
        optimizers = self.prepare_optimizers()
        training_data = self.prepare_data_for_training()
        output_dirs = model_utils.prepare_output_directory(self.config)
        callbacks = model_utils.setup_callbacks(self.config, output_dirs)

        metrics = model_utils.setup_metrics_callback(config=self.config,
                                                     model=self,
                                                     datasets=self.datasets,
                                                     training_data=training_data,
                                                     output_dirs=output_dirs,)

        # Training loop
        k_folds = len(training_data[0])
        for fold in range(k_folds):
            # Train a multi-task model (MTM)
            if len(optimizers) > 1:
                for epoch in range(self.config.epochs):
                    # Setup a progress bar
                    if k_folds > 1:
                        fold_and_epoch = (fold + 1, k_folds, epoch + 1, self.config.epochs)
                        print('Fold: {}/{}; Epoch: {}/{}'.format(*fold_and_epoch))
                    else:
                        print(f'Epoch: {epoch + 1}/{self.config.epochs}')

                    for model_idx in random.sample(range(0, len(optimizers)), len(optimizers)):
                        # Freeze all CRFs besides the one we are currently training
                        model_utils.freeze_output_layers(self.model, model_idx)

                        # Zero contribution to the loss from layers we aren't currently training
                        loss_weights = [1.0 if model_idx == i else 0.0
                                        for i, _ in enumerate(optimizers)]
                        self.compile(loss=crf_loss,
                                     optimizer=optimizers[model_idx],
                                     loss_weights=loss_weights)

                        # Get zeroed out targets for layers we aren't currently training
                        train_targets = \
                            [np.zeros_like(data[fold]['train']['y']) if i != model_idx else
                             data[fold]['train']['y'] for i, data in enumerate(training_data)]
                        valid_targets = \
                            [np.zeros_like(data[fold]['valid']['y']) if i != model_idx else
                             data[fold]['valid']['y'] for i, data in enumerate(training_data)]

                        callbacks_ = [cb[model_idx] for cb in callbacks] + [metrics[model_idx]]

                        validation_data = \
                            (training_data[model_idx][fold]['valid']['x'], valid_targets)

                        self.model.fit(
                            x=training_data[model_idx][fold]['train']['x'],
                            y=train_targets,
                            batch_size=self.config.batch_size,
                            callbacks=callbacks_,
                            validation_data=validation_data,
                            verbose=1,
                            # required to properly display current epoch
                            initial_epoch=epoch,
                            epochs=epoch + 1
                        )
            # Train a single-task model (STM)
            else:
                if k_folds > 1:
                    print(f'Fold: {fold + 1}/{self.config.k_folds}')

                callbacks_ = [cb[0] for cb in callbacks] + [metrics[0]]

                self.model.fit(
                    x=training_data[0][fold]['train']['x'],
                    y=training_data[0][fold]['train']['y'],
                    batch_size=self.config.batch_size,
                    callbacks=callbacks_,
                    validation_data=(training_data[0][fold]['valid']['x'],
                                     training_data[0][fold]['valid']['y']),
                    verbose=1,
                    epochs=self.config.epochs
                )

            # Clear and rebuild the model at end of each fold (except for the last fold)
            if fold < self.config.k_folds - 1:
                self.reset_model()
                optimizers = self.prepare_optimizers()

                for metric in metrics:
                    metric.on_fold_end()  # bumps internal fold counter

        return metrics

    def evaluate(self, training_data, partition='train', model_idx=-1):
        """Perform a prediction step using `self.model` on `training_data[partition]`.

        Performs prediction for the model at `self.model` and returns a two-tuple containing the
        true (gold) NER labels and corresponding predicted labels.

        Args:
            training_data (dict): Contains the inputs (at `training_data[partition]['x']`) and
                targets (at `training_data[partition]['y']`) used for evaluation.
            partition (str): Which partition to perform a prediction step on, must be one of
                'train', 'valid', 'test'.
            model_idx (int): Index to dataset in `self.datasets` that will be evaluated on.
                Defaults to -1.

        Returns:
            A two-tuple containing the gold NER label sequences and corresponding predicted labels.
        """
        dataset = self.datasets[model_idx]

        X, y = training_data[partition]['x'], training_data[partition]['y']

        # Gold labels
        y_true = y.argmax(axis=-1)
        y_pred = self.model.predict(X)

        # If multi-task model, take only predictions for output layer at model_idx
        if isinstance(y_pred, list):
            y_pred = y_pred[model_idx]

        y_pred = y_pred.argmax(axis=-1)

        # Mask [PAD] tokens
        y_true, y_pred = model_utils.mask_labels(y_true=y_true,
                                                 y_pred=y_pred,
                                                 label=dataset.type_to_idx['ent'][PAD])

        # Map predictions to tags
        y_true = [[dataset.idx_to_tag['ent'][idx] for idx in sent] for sent in y_true]
        y_pred = [[dataset.idx_to_tag['ent'][idx] for idx in sent] for sent in y_pred]

        return y_true, y_pred

    def predict(self, tokens):
        """Perform inference on tokenized and sentence segmented text.

        Using the model at `self.model`, performs inference on `tokens`, a list of lists
        containing tokenized sentences. The result of this inference is a list of lists, where the
        outer lists correspond to sentences in `tokens` and inner lists contains predicted labels
        for the corresponding tokens in `sents`.

        Args:
            tokens (list): List of lists containing tokenized sentences to annotate.

        Returns:
            If `self.models` has a single output layer (`not isinstance(self.model.output, list)`):
                A list of lists, containing the predicted labels for `tokens`.
            If `self.models` has multiple output layers (`isinstance(self.model.output, list)`):
                A list of lists of lists, containing the predicted labels for `tokens` from each
                output layer in `self.models`.
        """
        # These values are the same for all datasets
        idx_to_tag = self.datasets[0].idx_to_tag['ent']
        word_to_idx = self.datasets[0].type_to_idx['word']
        char_to_idx = self.datasets[0].type_to_idx['char']

        word_idx_seq = Preprocessor.get_type_idx_sequence(tokens, word_to_idx, type_='word')
        char_idx_seq = Preprocessor.get_type_idx_sequence(tokens, char_to_idx, type_='char')
        model_input = [word_idx_seq, char_idx_seq]

        # Actual prediction happens here
        y_preds = self.model.predict(model_input)

        # If y_preds is not a list, this is a STM. Add dummy first dimension
        y_preds = [y_preds] if not isinstance(y_preds, list) else y_preds

        # Mask [PAD] tokens
        y_preds_masked = []
        for y_pred in y_preds:
            y_pred = np.argmax(y_pred, axis=-1)
            _, y_pred = model_utils.mask_labels(y_true=word_idx_seq, y_pred=y_pred, label=PAD_VALUE)
            y_preds_masked.append([[idx_to_tag[idx] for idx in sent] for sent in y_pred])

        # If STM, return only a list of lists
        if len(y_preds_masked) == 1:
            y_preds_masked = y_preds_masked[0]

        return y_preds_masked

    def prepare_for_transfer(self, datasets):
        """Prepares the BiLSTM-CRF for transfer learning by recreating its last layer(s).

        Prepares the BiLSTM-CRF model at `self.model` for transfer learning by removing the
        CRF classifier(s) and replacing them with un-trained CRF classifiers of the appropriate size
        (i.e. number of units equal to number of output tags) for the target datasets `datasets`.

        Args:
            datasets (list): A list of `Dataset` objects.

        Returns:
            `self.model` after replacing its output layers with freshly initialized output layers,
            one per dataset in `datasets`.
        """
        # remove all existing output layers
        n_output_layers = 1 if not isinstance(self.model.output, list) else len(self.model.output)
        _ = [self.model.layers.pop() for _ in range(n_output_layers)]

        # get new output layers, one per target dataset
        new_outputs = []
        for i, dataset in enumerate(datasets):
            output = CRF(len(dataset.type_to_idx['ent']), name=f'crf_{i}')
            output = output(self.model.layers[-1].output)

            new_outputs.append(output)

        # create a new model, with new output layers, one per target dataset
        self.model = Model(inputs=self.model.input, outputs=new_outputs)

        # replace datasets linked to this model with target datasets
        self.datasets = datasets

        self.compile()

        return self.model

    def prepare_optimizers(self):
        """Returns a list of Keras optimizers, one per output layer in `self.model`.

        For each output layer in `self.model`, creates an optmizer based on the given config at
        `self.config`. For a single-task model, the returned list will be of length 1. For
        a multi-task model, the returned list will be of length `len(self.model.ouput)`.

        Returns:
            A list of Keras optimizers initiated from the given config at `self.config`.
        """
        optimizer_args = (self.config.optimizer,
                          self.config.learning_rate,
                          self.config.decay,
                          self.config.grad_norm)

        outputs = self.model.output if isinstance(self.model.output, list) else [self.model.output]
        optimizers = [model_utils.get_keras_optimizer(*optimizer_args)
                      for _, _ in enumerate(outputs)]

        return optimizers
