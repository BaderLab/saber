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

# https://stackoverflow.com/questions/48615003/multi-task-learning-in-keras
# https://machinelearningmastery.com/keras-functional-api-deep-learning/

class MultiTaskLSTMCRF(object):
    """ Implements a MT biLSTM-CRF for NER and TWI using Keras. """

    def __init__(self, model_specifications):
        # Grab the specs we need to build the model. Keys correspond to names of
        # attributes of Dataset and SequenceProcessingModel classes.
        self.max_seq_len = model_specifications['max_seq_len']
        self.activation_function = model_specifications['activation_function']
        self.batch_size = model_specifications['batch_size']
        self.dropout_rate = model_specifications['dropout_rate']
        self.maximum_number_of_epochs = model_specifications['maximum_number_of_epochs']
        self.num_of_ds = len(model_specifications['dataset_text_folder'])
        self.token_embedding_matrix = model_specifications['token_embedding_matrix']
        self.freeze_token_embeddings = model_specifications['freeze_token_embeddings']
        self.learning_rate = model_specifications['learning_rate']
        self.optimizer = model_specifications['optimizer']

        # model(s) tied to this instance
        self.ds = model_specifications['ds']
        self.model = []
        self.crf = []

    def specify_(self):
        """ Specifies a multi-task bidirectional LSTM-CRF using Keras.

        Implements a hybrid long short-term memory network-condition random field
        (LSTM-CRF) multi-task model.

        Returns:
            model: a keras model, excluding including crf layer
            crf: a crf layer implemented in keras.contrib
        """
        ## TOKEN EMBEDDING LAYER
        # if specified, load pre-trained token embeddings otherwise initialize
        # randomly
        if self.token_embedding_matrix is not None:
            # plus 1 because of '0' word.
            shared_token_emb = Embedding(
                # input dimension size of all word types shared across datasets
                input_dim=len(self.ds[0].word_type_to_idx) + 1,
                output_dim=self.token_embedding_matrix.shape[1],
                weights=[self.token_embedding_matrix],
                input_length=self.max_seq_len,
                mask_zero=True,
                trainable=(not self.freeze_token_embeddings))
        else:
            shared_token_emb = Embedding(
                input_dim=len(self.ds[0].word_type_to_idx) + 1,
                output_dim=100,
                input_length=self.max_seq_len,
                mask_zero=True)

        ## TOKEN BILSTM LAYER
        shared_token_bisltm = Bidirectional(LSTM(
            units=100,
            return_sequences=True,
            recurrent_dropout=self.dropout_rate))

        ## FULLY CONNECTED LAYER
        shared_dense = TimeDistributed(Dense(
            units=100,
            activation=self.activation_function))

        for ds in self.ds:
            input_layer = Input(shape=(self.max_seq_len,))
            model = shared_token_emb(input_layer)
            model = shared_token_bisltm(model)
            model = shared_dense(model)
            crf = CRF(ds.tag_type_count)
            output_layer = crf(model)

            self.model.append(Model(inputs=input_layer, outputs=output_layer))
            self.crf.append(crf)

            input_layer = None
            model = None
            crf = None
            output_layer = None

        return self.model

    def compile_(self):
        """ Compiles a multi-task biLSTM-CRF for NER using Keras. """
        for model, crf in zip(self.model, self.crf):
            compile_model(model=model,
                          learning_rate=self.learning_rate,
                          optimizer=self.optimizer,
                          loss_function=crf.loss_function,
                          metrics=crf.accuracy)

    def fit_(self, dataset):

        '''
        self.model[0].fit(x=dataset[0].train_word_idx_sequence,
                          y=dataset[0].train_tag_idx_sequence,
                          batch_size=self.batch_size,
                          epochs=1,
                          validation_split=0.1,
                          verbose=1)
        '''


        for epoch in range(self.maximum_number_of_epochs):
            for ds, model in zip(self.ds, self.model):
                model.fit(x=ds.train_word_idx_sequence,
                          y=ds.train_tag_idx_sequence,
                          batch_size=self.batch_size,
                          epochs=1,
                          initial_epoch=epoch
                          validation_split=0.1,
                          verbose=1)




        '''

        X_train_1 = dataset[0].train_word_idx_sequence
        X_train_2 = dataset[1].train_word_idx_sequence

        y_train_1 = dataset[0].train_tag_idx_sequence
        y_train_2 = dataset[1].train_tag_idx_sequence

        # check that input/label shapes make sense
        assert X_train_1.shape[0] == y_train_1.shape[0]
        assert X_train_1.shape[1] == y_train_1.shape[1]

        assert X_train_2.shape[0] == y_train_2.shape[0]
        assert X_train_2.shape[1] == y_train_2.shape[1]

        assert y_train_1.shape[-1] == y_train_2.shape[-1]


        train_history = self.model.fit(x=X_train_1,
                                       y=[y_train_1, y_train_2],
                                       batch_size=self.batch_size,
                                       epochs=self.maximum_number_of_epochs,
                                       validation_split=0.1,
                                       verbose=1)

        return train_history
        '''

    def _specify_shared_model(self, shared_layers):
        """ A helper function for specify the model architecture.

        Because of how models with shared layers are created in Keras, this
        function is neccecary to properly share some layers. Input and
        output layers are specified in this method, but the shared layers
        are passed in as argument.

        Returns:
            input_: a new input layer of shape self.max_seq_len, None
            output_: a new output
            crf: a crf layer object from Keras contrib
        """
        # the input layer must be of fixed length because of the CRF output layer

        input_layer = shared_layers['input_layer']
        shared_token_emb = shared_layers['shared_token_emb'](input_layer)
        shared_token_bisltm = shared_layers['shared_token_bisltm'](shared_token_emb)
        shared_dense = shared_layers['shared_dense'](shared_token_bisltm)
        crf = CRF(self.tag_type_count)
        output_ = crf(shared_dense)

        return output_, crf
