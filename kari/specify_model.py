from keras import optimizers
from keras.models import Model
from keras.models import Input
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras_contrib.layers import CRF

# there is a bunch of preprocessing steps I need to perform.
# 1: I need to append a "sentence number" to each "sentence group" (not entirely sure if this is necc.)
# 2: need to get POS tags (also not necc. but would be good to do)

def specify_LSTM_CRF_(model_specifications):
    """
    """
    ## SPECIFY
    # Grab the specs we need to build the model. Keys correspond to names of
    # attributes of Dataset and SequenceProcessingModel classes.
    max_seq_len = model_specifications['max_seq_len']
    word_type_count = model_specifications['word_type_count']
    tag_type_count = model_specifications['tag_type_count']
    activation_function = model_specifications['activation_function']
    dropout_rate = model_specifications['dropout_rate']

    ## BUILD
    # build the model
    input_ = Input(shape=(max_seq_len,))
    # plus 1 because of '0' word.
    model = Embedding(input_dim=word_type_count + 1, output_dim=20,
                      input_length=max_seq_len, mask_zero=True)(input_)
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=dropout_rate))(model)
    model = TimeDistributed(Dense(50, activation=activation_function))(model)

    crf = CRF(tag_type_count)
    out = crf(model)

    model = Model(input_, out)

    return model, crf

def compile_LSTM_CRF_(model_specifications, model, crf):
    """
    """
    # Grab the specs we need to build the model. Keys correspond to names of
    # attributes of SequenceProcessingModel classe.
    learning_rate = model_specifications['learning_rate']
    optimizer = model_specifications['optimizer']

    ## COMPILE
    if optimizer == 'sgd':
        optimizer_ = optimizers.SGD(lr=learning_rate)

    model.compile(optimizer=optimizer_, loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()
