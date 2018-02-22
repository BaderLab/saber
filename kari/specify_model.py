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

# TODO (johngiorgi) set up parameter for embedding output dimension
# TODO (johngiorgi) considering organizing these into classes

class SimpleLSTMCRF(object):
    """ Implements a bidirectional LSTM-CRF for NER using Keras. """

    def __init__(self, model_specifications):
        # Grab the specs we need to build the model. Keys correspond to names of
        # attributes of Dataset and SequenceProcessingModel classes.
        self.max_seq_len = model_specifications['max_seq_len']
        self.word_type_count = model_specifications['word_type_count']
        self.tag_type_count = model_specifications['tag_type_count']
        self.activation_function = model_specifications['activation_function']
        self.dropout_rate = model_specifications['dropout_rate']
        self.token_embedding_matrix = model_specifications['token_embedding_matrix']
        self.freeze_token_embeddings = model_specifications['freeze_token_embeddings']
        self.learning_rate = model_specifications['learning_rate']
        self.optimizer = model_specifications['optimizer']

        self.model = None
        self.crf = None

    def specify_(self):
        """ Specifies a bidirectional LSTM-CRF for NER using Keras.

        Implements a hybrid long short-term memory network-condition random field
        (LSTM-CRF) for the task of NER.

        Args:
            model_specifications: a dictionary containing model hyperparameters.

        Returns:
            model: a keras model, excluding including crf layer
            crf: a crf layer implemented in keras.contrib
        """
        ## SPECIFY

        # the input layer must be of fixed length because of the CRF output layer
        input_ = Input(shape=(self.max_seq_len,))

        # token embedding layer
        # if specified, load pre-trained token embeddings otherwise initialize
        # randomly
        if self.token_embedding_matrix is not None:
            # plus 1 because of '0' word.
            model = Embedding(input_dim=self.word_type_count + 1,
                              output_dim=self.token_embedding_matrix.shape[1],
                              weights=[self.token_embedding_matrix],
                              input_length=self.max_seq_len,
                              mask_zero=True,
                              trainable= not self.freeze_token_embeddings)(input_)
        else:
            model = Embedding(input_dim=self.word_type_count + 1,
                              output_dim=100,
                              input_length=self.max_seq_len,
                              mask_zero=True)(input_)
        # token LSTM layer
        model = Bidirectional(LSTM(units=100, return_sequences=True,
                                   recurrent_dropout=self.dropout_rate))(model)
        # fully connected layer
        model = TimeDistributed(Dense(100, activation=self.activation_function))(model)
        # sequence optimizing output layer (CRF)
        crf = CRF(self.tag_type_count)
        out = crf(model)

        # fully specified model
        model = Model(input_, out)

        # update class attributes
        self.model, self.crf = model, crf

    def compile_(self):
        """ Compiles a bidirectional LSTM-CRF for NER using Keras.
        """
        compile_model(learning_rate = self.learning_rate,
                      optimizer = self.optimizer,
                      model = self.model,
                      loss_function = self.crf.loss_function,
                      metrics = self.crf.accuracy)
