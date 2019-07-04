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

from .. import constants
from ..preprocessor import Preprocessor
from ..utils import data_utils
from ..utils import model_utils
from .base_model import BaseKerasModel

LOGGER = logging.getLogger(__name__)


class BiLSTMCRF(BaseKerasModel):
    """A Keras implementation of a Multi-task BiLSTM-CRF for sequence labeling.

    A BiLSTM-CRF for NER implementation in Keras. Supports multi-task learning by default, just pass
    multiple Dataset objects via `datasets` to the constructor and the model will share the
    parameters of all layers, except for the final output layer, across all datasets, where each
    dataset represents a sequence labelling task.

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
        """Specifies a BiLSTM-CRF for sequence tagging using Keras.

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
        char_BiLSTM = TimeDistributed(Bidirectional(LSTM(constants.UNITS_CHAR_LSTM // 2)),
                                      name='character_BiLSTM')
        # Word-level BiLSTM
        word_BiLSTM_1 = Bidirectional(LSTM(units=constants.UNITS_WORD_LSTM // 2,
                                           return_sequences=True,
                                           dropout=self.config.dropout_rate['input'],
                                           recurrent_dropout=self.config.dropout_rate['recurrent']),
                                      name="word_BiLSTM_1")
        word_BiLSTM_2 = Bidirectional(LSTM(units=constants.UNITS_WORD_LSTM // 2,
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

    def train(self):
        """Co-ordinates the training of the Keras model `self.model`.

        Co-ordinates the training of the Keras model `self.model`. Minimally expects a train
        partition and one or both of valid and test partitions to be supplied in the Dataset objects
        at `self.datasets`.
        """
        # Need to explicitly get one optimizer per output layer (if more than one)
        optimizers = self.prepare_optimizers()

        # Gather everything we need to run a training session
        training_data = self.prepare_data_for_training()
        output_dir = model_utils.prepare_output_directory(self.config)
        callbacks = model_utils.setup_callbacks(self.config, output_dir)

        def train_valid_test(training_data, output_dir, callbacks, optimizers=None):
            # Use 10% of train data as validation data if no validation data provided
            if training_data[0]['x_valid'] is None:
                training_data = data_utils.collect_valid_data(training_data)

            # Get list of Keras Callback objects for computing/storing metrics
            metrics = model_utils.setup_metrics_callback(model=self,
                                                         config=self.config,
                                                         datasets=self.datasets,
                                                         training_data=training_data,
                                                         output_dir=output_dir,)

            # Train a multi-task model (MTM)
            if len(optimizers) > 1:
                for epoch in range(self.config.epochs):
                    print(f'Global epoch: {epoch + 1}/{self.config.epochs}')

                    for model_idx in random.sample(range(0, len(optimizers)), len(optimizers)):

                        # Freeze all CRFs besides the one we are currently training
                        model_utils.freeze_output_layers(self.model, model_idx)

                        # Zero the contribution to the loss from layers we aren't currently training
                        loss_weights = [1.0 if model_idx == i else 0.0
                                        for i, _ in enumerate(optimizers)]
                        self.compile(loss=crf_loss, optimizer=optimizers[model_idx],
                                     loss_weights=loss_weights)

                        # Get zeroed out targets for layers we aren't currently training
                        train_targets, valid_targets = self.model_utils.get_targets(training_data,
                                                                                    model_idx)

                        callbacks_ = [cb[model_idx] for cb in callbacks] + [metrics[model_idx]]
                        validation_data = (training_data[model_idx]['x_valid'], valid_targets)

                        self.model.fit(x=training_data[model_idx]['x_train'],
                                       y=train_targets,
                                       batch_size=self.config.batch_size,
                                       callbacks=callbacks_,
                                       validation_data=validation_data,
                                       verbose=1,
                                       # required for Keras to properly display current epoch
                                       initial_epoch=epoch,
                                       epochs=epoch + 1)
            # Train a single-task model (STM)
            else:
                callbacks_ = [cb[0] for cb in callbacks] + [metrics[0]]

                self.model.fit(x=training_data[0]['x_train'],
                               y=training_data[0]['y_train'],
                               batch_size=self.config.batch_size,
                               callbacks=callbacks_,
                               validation_data=(training_data[0]['x_valid'],
                                                training_data[0]['y_valid']),
                               verbose=1,
                               epochs=self.config.epochs)

        def cross_validation(training_data, output_dir, callbacks, optimizers=None):
            # Get the train/valid partitioned data for all datasets and all folds
            training_data = data_utils.collect_cv_data(training_data, self.config.k_folds)

            # Training loop
            for fold in range(self.config.k_folds):
                # get list of Keras Callback objects for computing/storing metrics
                metrics = model_utils.setup_metrics_callback(model=self,
                                                             datasets=self.datasets,
                                                             config=self.config,
                                                             training_data=training_data,
                                                             output_dir=output_dir,
                                                             fold=fold)

                # Train a multi-task model (MTM)
                if optimizers is not None and len(optimizers) > 1:
                    for epoch in range(self.config.epochs):
                        train_info = (fold + 1, self.config.k_folds, epoch + 1, self.config.epochs)
                        print('Fold: {}/{}; Global epoch: {}/{}'.format(*train_info))

                        for model_idx in random.sample(range(0, len(optimizers)), len(optimizers)):
                            # Freeze all CRFs besides the one we are currently training
                            model_utils.freeze_output_layers(self.model, model_idx)

                            # Zero contribution to the loss from layers we aren't currently training
                            loss_weights = [1.0 if model_idx == i else 0.0
                                            for i, _ in enumerate(optimizers)]
                            self.compile(loss=crf_loss, optimizer=optimizers[model_idx],
                                         loss_weights=loss_weights)

                            # Get zeroed out targets for layers we aren't currently training
                            train_targets, valid_targets = \
                                model_utils.get_targets(training_data, model_idx, fold)

                            callbacks_ = [cb[model_idx] for cb in callbacks] + [metrics[model_idx]]
                            validation_data = (training_data[model_idx][fold]['x_valid'],
                                               valid_targets)

                            self.model.fit(x=training_data[model_idx][fold]['x_train'],
                                           y=train_targets,
                                           batch_size=self.config.batch_size,
                                           callbacks=callbacks_,
                                           validation_data=validation_data,
                                           verbose=1,
                                           # required to properly display current epoch
                                           initial_epoch=epoch,
                                           epochs=epoch + 1)
                # Train a single-task model (STM)
                else:
                    print(f'Fold: {fold + 1}/{self.config.k_folds}')

                    callbacks_ = [cb[0] for cb in callbacks] + [metrics[0]]

                    self.model.fit(x=training_data[0][fold]['x_train'],
                                   y=training_data[0][fold]['y_train'],
                                   batch_size=self.config.batch_size,
                                   callbacks=callbacks_,
                                   validation_data=(training_data[0][fold]['x_valid'],
                                                    training_data[0][fold]['y_valid']),
                                   verbose=1,
                                   epochs=self.config.epochs)

                # Clear and rebuild the model at end of each fold (except for the last fold)
                if fold < self.config.k_folds - 1:
                    self.reset_model()
                    optimizers = self.prepare_optimizers()

        # TODO: User should be allowed to overwrite this
        if training_data[0]['x_valid'] is not None or training_data[0]['x_test'] is not None:
            print('Using train/test/valid strategy...')
            LOGGER.info('Used a train/test/valid strategy for training')
            train_valid_test(training_data, output_dir, callbacks, optimizers)
        else:
            print(f'Using {self.config.k_folds}-fold cross-validation strategy...')
            LOGGER.info('Used %s-fold cross-validation strategy for training', self.config.k_folds)
            cross_validation(training_data, output_dir, callbacks, optimizers)

    def evaluate(self, training_data, model_idx=-1, partition='train'):
        """Get `y_true` and `y_pred` for given inputs and targets in `training_data`.

        Performs prediction for the model at `self.models[model_idx]` and returns a 2-tuple
        containing the true (gold) labels and the predicted labels, where labels are integers
        corresponding to mapping at `self.idx_to_tag`. Inputs are given at
        `training_data[x_partition]` and gold labels at `training_data[y_partition]`.

        Args:
            training_data (dict): Contains the data (at key `x_partition`) and targets
                (at key `y_partition`) for each partition: 'train', 'valid' and 'test'.
            partition (str): Which partition to perform a prediction step on, must be one of
                'train', 'valid', 'test'.

        Returns:
            A two-tuple containing the gold label integer sequences and the predicted integer label
            sequences.
        """
        dataset = self.datasets[model_idx]

        X, y = training_data[f'x_{partition}'], training_data[f'y_{partition}']

        # Gold labels
        y_true = y.argmax(axis=-1).ravel()
        y_pred = self.model.predict(X)

        # If multi-task model, take only predictions for output layer at model_idx
        if isinstance(y_pred, list):
            y_pred = y_pred[model_idx]

        y_pred = y_pred.argmax(axis=-1).ravel()

        # Mask [PAD] tokens
        y_true, y_pred = \
            model_utils.mask_labels(y_true, y_pred, dataset.type_to_idx['ent'][constants.PAD])

        return y_true, y_pred

    def predict(self, tokens):
        """Performs inference on tokenized text and returns the predicted labels.

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
        # Prepare data for input to model
        word_to_idx = self.datasets[0].type_to_idx['word']
        char_to_idx = self.datasets[0].type_to_idx['char']  # These are the same for all datasets

        word_idx_seq = Preprocessor.get_type_idx_sequence(seq=tokens,
                                                          type_to_idx=word_to_idx,
                                                          type_='word')
        char_idx_seq = Preprocessor.get_type_idx_sequence(seq=tokens,
                                                          type_to_idx=char_to_idx,
                                                          type_='char')
        model_input = [word_idx_seq, char_idx_seq]

        # Actual prediction happens here
        y_preds = self.model.predict(model_input)
        y_preds = np.argmax(y_preds, axis=-1)

        # If y_preds is not a list, this is a STM. Add dummy first dimension
        if not isinstance(y_preds, list):
            y_preds = [y_preds]

        # Mask [PAD] tokens
        y_preds_masked = []
        for y_pred, dataset in zip(y_preds, self.datasets):
            _, y_pred = model_utils.mask_labels(y_true=word_idx_seq,
                                                y_pred=y_pred,
                                                label=constants.PAD_VALUE)

            y_preds_masked.append([[dataset.idx_to_tag['ent'][idx] for idx in sent]
                                   for sent in y_pred])

        # If STM, return only a list of lists
        if len(y_preds_masked) == 1:
            y_preds_masked = y_preds_masked[0]

        return y_preds_masked

    def prepare_data_for_training(self):
        """Returns a list containing the training data for each dataset at `self.datasets`.

        For each dataset at `self.datasets`, collects the data to be used for training.
        Each dataset is represented by a dictionary, where the keys 'x_<partition>' and
        'y_<partition>' contain the inputs and targets for each partition 'train', 'valid', and
        'test'.

        Returns:
            A list of dictionaries containing the training data for each dataset at `self.datasets`.
        """
        training_data = []
        for ds in self.datasets:
            # collect train data, must be provided
            x_train = [ds.idx_seq['train']['word'], ds.idx_seq['train']['char']]
            y_train = to_categorical(ds.idx_seq['train']['ent'])
            # collect valid and test data, may not be provided
            x_valid, y_valid, x_test, y_test = None, None, None, None
            if ds.idx_seq['valid'] is not None:
                x_valid = [ds.idx_seq['valid']['word'], ds.idx_seq['valid']['char']]
                y_valid = to_categorical(ds.idx_seq['valid']['ent'])
            if ds.idx_seq['test'] is not None:
                x_test = [ds.idx_seq['test']['word'], ds.idx_seq['test']['char']]
                y_test = to_categorical(ds.idx_seq['test']['ent'])

            training_data.append({'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid,
                                  'y_valid': y_valid, 'x_test': x_test, 'y_test': y_test})

        return training_data

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
