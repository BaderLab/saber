from keras import optimizers
from keras.models import Model
from keras.models import Input
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras_contrib.layers.crf import CRF

from utils_models import compile_model

# TODO (johngiorgi): setup gradient clipping

class SimpleLSTMCRF(object):
    """ Implements a bidirectional LSTM-CRF for NER using Keras. """

    def __init__(self, model_specifications):
        # Grab the specs we need to build the model. Keys correspond to names of
        # attributes of Dataset and SequenceProcessingModel classes.
        self.activation_function = model_specifications['activation_function']
        self.batch_size = model_specifications['batch_size']
        self.dropout_rate = model_specifications['dropout_rate']
        self.freeze_token_embeddings = model_specifications['freeze_token_embeddings']
        self.gradient_clipping_value = model_specifications['gradient_clipping_value']
        self.learning_rate = model_specifications['learning_rate']
        self.maximum_number_of_epochs = model_specifications['maximum_number_of_epochs']
        self.optimizer = model_specifications['optimizer']
        self.token_embedding_matrix = model_specifications['token_embedding_matrix']
        self.token_embedding_dimension = model_specifications['token_embedding_dimension']
        self.max_seq_len = model_specifications['max_seq_len']

        # dataset tied to this instance
        self.ds = model_specifications['ds']
        # model tied to this instance
        self.model = None
        self.crf = None

    def specify_(self):
        """ Specifies a bidirectional LSTM-CRF for sequence tagging using Keras.

        Implements a hybrid long short-term memory network-condition random field
        (LSTM-CRF) for sequence tagging.

        Returns:
            model: a keras model, excluding crf layer
            crf: a crf layer implemented using keras.contrib
        """
        ## INPUT LAYER
        # the input layer must be of fixed length because of the CRF output layer
        input_layer = Input(shape=(self.max_seq_len,))

        ## TOKEN EMBEDDING LAYER
        # if specified, load pre-trained token embeddings otherwise initialize
        # randomly
        if self.token_embedding_matrix is not None:
            # plus 1 because of '0' word.
            token_embed = Embedding(
                input_dim=self.ds[0].word_type_count + 1,
                output_dim=self.token_embedding_matrix.shape[1],
                weights=[self.token_embedding_matrix],
                input_length=self.max_seq_len,
                mask_zero=True,
                trainable=(not self.freeze_token_embeddings))(input_layer)
        else:
            token_embed = Embedding(
                input_dim=self.ds[0].word_type_count + 1,
                output_dim=self.token_embedding_dimension,
                input_length=self.max_seq_len,
                mask_zero=True)(input_)
        ## TOKEN BILSTM LAYER
        token_bilstm = Bidirectional(LSTM(
            units=100,
            return_sequences=True,
            recurrent_dropout=self.dropout_rate))(token_embed)
        ## FULLY CONNECTED LAYER
        fully_connected = TimeDistributed(Dense(
            units=100,
            activation=self.activation_function))(token_bilstm)

        ## SEQUENCE OPTIMIZING LAYER (CRF)
        crf = CRF(self.ds[0].tag_type_count)
        output_layer = crf(fully_connected)

        # fully specified model
        model = Model(inputs=input_layer, outputs=output_layer)

        # update class attributes
        self.model, self.crf = model, crf

    def compile_(self):
        """ Compiles a bidirectional LSTM-CRF for for sequence tagging using
        Keras. """
        compile_model(learning_rate = self.learning_rate,
                      optimizer = self.optimizer,
                      model = self.model,
                      loss_function = self.crf.loss_function,
                      metrics = self.crf.accuracy)

    def fit_(self):
        """ Fits a bidirectional LSTM-CRF for for sequence tagging using
        Keras. """
        X_train = self.ds[0].train_word_idx_sequence
        y_train = self.ds[0].train_tag_idx_sequence

        train_history = self.model.fit(x=X_train,
                                       y=y_train,
                                       batch_size=self.batch_size,
                                       epochs=self.maximum_number_of_epochs,
                                       validation_split=0.1,
                                       # callbacks = [checkpointer, metrics],
                                       verbose=1)
        return train_history
