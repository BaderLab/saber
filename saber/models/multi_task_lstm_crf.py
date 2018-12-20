"""Contains the Multi-task BiLSTM-CRF (MT-BILSTM-CRF) Keras model for squence labelling.
"""
import logging

import tensorflow as tf
from keras.layers import (LSTM, Bidirectional, Concatenate, Dense, Dropout,
                          Embedding, SpatialDropout1D, TimeDistributed)
from keras.models import Input, Model, model_from_json
from keras.utils import multi_gpu_model
from keras_contrib.layers.crf import CRF
from keras_contrib.losses.crf_losses import crf_loss

from .. import constants
from .base_model import BaseKerasModel

LOGGER = logging.getLogger(__name__)

class MultiTaskLSTMCRF(BaseKerasModel):
    """A Keras implementation of a BiLSTM-CRF for sequence labeling.

    A BiLSTM-CRF for NER implementation in Keras. Supports multi-task learning by default, just pass
    multiple Dataset objects via `datasets` to the constructor and the model will share the
    parameters of all layers, except for the final output layer, across all datasets, where each
    dataset represents a sequence labelling task.

    Args:
        config (Config): Contains a set of harmonzied arguments provided in a *.ini file and,
            optionally, from the command line.
        datasets (list): A list of Dataset objects.
        embeddings (numpy.ndarray): A numpy array where ith row contains the vector embedding for
            the ith word type.

    References:
        - Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
          "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
          https://arxiv.org/abs/1603.01360
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        super().__init__(config, datasets, embeddings, **kwargs)

    def load(self, weights_filepath, model_filepath):
        """Load a model from disk.

        Loads a model from disk by loading its architecture from a json file at `model_filepath`
        and its weights from a hdf5 file at `model_filepath`.

        Args:
            weights_filepath (str): Filepath to the models wieghts (.hdf5 file).
            model_filepath (str): Filepath to the models architecture (.json file).
        """
        with open(model_filepath) as f:
            model = model_from_json(f.read(), custom_objects={'CRF': CRF})
            model.load_weights(weights_filepath)
            self.models.append(model)

    def specify(self):
        """Specifies a multi-task BiLSTM-CRF for sequence tagging using Keras.

        Implements a hybrid bidirectional long short-term memory network-condition random
        field (BiLSTM-CRF) multi-task model for sequence tagging.
        """
        # specify any shared layers outside the for loop
        # word-level embedding layer
        if self.embeddings is None:
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
        # character-level embedding layer
        char_embeddings = Embedding(input_dim=len(self.datasets[0].type_to_idx['char']) + 1,
                                    output_dim=self.config.char_embed_dim,
                                    mask_zero=True,
                                    name="char_embedding_layer")
        # char-level BiLSTM
        char_BiLSTM = TimeDistributed(Bidirectional(LSTM(constants.UNITS_CHAR_LSTM // 2)),
                                      name='character_BiLSTM')
        # word-level BiLSTM
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

        # get all unique tag types across all datasets
        all_tags = [ds.type_to_idx['tag'] for ds in self.datasets]
        all_tags = set(x for l in all_tags for x in l)

        # feedforward before CRF, maps each time step to a vector
        dense_layer = TimeDistributed(Dense(len(all_tags), activation=self.config.activation),
                                      name='dense_layer')

        # specify model, taking into account the shared layers
        for dataset in self.datasets:
            # word-level embedding
            word_ids = Input(shape=(None, ), dtype='int32', name='word_id_inputs')
            word_embed = word_embeddings(word_ids)

            # character-level embedding
            char_ids = Input(shape=(None, None), dtype='int32', name='char_id_inputs')
            char_embed = char_embeddings(char_ids)

            # character-level BiLSTM + dropout. Spatial dropout applies the same dropout mask to all
            # timesteps which is necessary to implement variational dropout
            # (https://arxiv.org/pdf/1512.05287.pdf)
            char_embed = char_BiLSTM(char_embed)
            if self.config.variational_dropout:
                LOGGER.info('Used variational dropout')
                char_embed = SpatialDropout1D(self.config.dropout_rate['output'])(char_embed)

            # concatenate word- and char-level embeddings + dropout
            model = Concatenate()([word_embed, char_embed])
            model = Dropout(self.config.dropout_rate['output'])(model)

            # word-level BiLSTM + dropout
            model = word_BiLSTM_1(model)
            if self.config.variational_dropout:
                model = SpatialDropout1D(self.config.dropout_rate['output'])(model)
            model = word_BiLSTM_2(model)
            if self.config.variational_dropout:
                model = SpatialDropout1D(self.config.dropout_rate['output'])(model)

            # feedforward before CRF
            model = dense_layer(model)

            # CRF output layer
            crf = CRF(len(dataset.type_to_idx['tag']), name='crf_classifier')
            output_layer = crf(model)

            # fully specified model
            # https://github.com/keras-team/keras/blob/bf1378f39d02b7d0b53ece5458f9275ac8208046/keras/utils/multi_gpu_utils.py
            with tf.device('/cpu:0'):
                model = Model(inputs=[word_ids, char_ids], outputs=[output_layer])
            self.models.append(model)

    def compile(self):
        """Compiles the BiLSTM-CRF.

        Compiles the Keras model(s) at `self.models`. If multiple GPUs are detected, a model
        capable of training on all of them is compiled.
        """
        for i in range(len(self.models)):
            # parallize the model if multiple GPUs are available
            # https://github.com/keras-team/keras/pull/9226
            # awfully bad practice but this was the example given by Keras documentation
            try:
                self.models[i] = multi_gpu_model(self.models[i])
                LOGGER.info('Compiling the model on multiple GPUs')
            except:
                LOGGER.info('Compiling the model on a single CPU or GPU')

            self._compile(model=self.models[i],
                          loss_function=crf_loss,
                          optimizer=self.config.optimizer,
                          lr=self.config.learning_rate,
                          decay=self.config.decay,
                          clipnorm=self.config.grad_norm)

    def prepare_for_transfer(self, datasets):
        """Prepares the BiLSTM-CRF for transfer learning by recreating its last layer.

        Prepares the BiLSTM-CRF model(s) at `self.models` for transfer learning by removing their
        CRF classifiers and replacing them with un-trained CRF classifiers of the appropriate size
        (i.e. number of units equal to number of output tags) for the target datasets.

        References:
        - https://stackoverflow.com/questions/41378461/how-to-use-models-from-keras-applications-for-transfer-learnig/41386444#41386444
        """
        self.datasets = datasets # replace with target datasets
        models, self.models = self.models, [] # wipe models

        for dataset, model in zip(self.datasets, models):
            # remove the old CRF classifier and define a new one
            model.layers.pop()
            new_crf = CRF(len(dataset.type_to_idx['tag']), name='target_crf_classifier')
            # create the new model
            new_input = model.input
            new_output = new_crf(model.layers[-1].output)
            self.models.append(Model(new_input, new_output))

        self.compile()
