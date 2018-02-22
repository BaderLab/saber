from keras import optimizers
from keras.models import Model
from keras.models import Input

# TODO (johngiorgi) set up parameter for embedding output dimension
# TODO (johngiorgi) considering organizing these into classes

def specify_LSTM_CRF(model_specifications):
    """ Specifies a bidirectional LSTM-CRF for NER using Keras.

    Implements a hybrid long short-term memory network-condition random field
    (LSTM-CRF) for the task of NER.

    Args:
        model_specifications: a dictionary containing model hyperparameters.

    Returns:
        model: a keras model, excluding including crf layer
        crf: a crf layer implemented in keras.contrib
    """
    # imports neccecary for specifying model
    from keras.layers import LSTM
    from keras.layers import Embedding
    from keras.layers import Dense
    from keras.layers import TimeDistributed
    from keras.layers import Dropout
    from keras.layers import Bidirectional

    from keras_contrib.layers.crf import CRF
    ## SPECIFY
    # Grab the specs we need to build the model. Keys correspond to names of
    # attributes of Dataset and SequenceProcessingModel classes.
    max_seq_len = model_specifications['max_seq_len']
    word_type_count = model_specifications['word_type_count']
    tag_type_count = model_specifications['tag_type_count']
    activation_function = model_specifications['activation_function']
    dropout_rate = model_specifications['dropout_rate']

    token_embedding_matrix = model_specifications['token_embedding_matrix']
    freeze_token_embeddings = model_specifications['freeze_token_embeddings']

    ## BUILD
    # the input layer must be of fixed length because of the CRF output layer
    input_ = Input(shape=(max_seq_len,))

    # token embedding layer
    # if specified, load pre-trained token embeddings otherwise initialize
    # randomly
    if token_embedding_matrix is not None:
        # plus 1 because of '0' word.
        model = Embedding(input_dim=word_type_count + 1,
                          output_dim=token_embedding_matrix.shape[1],
                          weights=[token_embedding_matrix],
                          input_length=max_seq_len,
                          mask_zero=True,
                          trainable=(not freeze_token_embeddings))(input_)
    else:
        model = Embedding(input_dim=word_type_count + 1,
                          output_dim=100,
                          input_length=max_seq_len,
                          mask_zero=True)(input_)
    # token LSTM layer
    model = Bidirectional(LSTM(units=100, return_sequences=True,
                               recurrent_dropout=dropout_rate))(model)
    # fully connected layer
    model = TimeDistributed(Dense(100, activation=activation_function))(model)
    # sequence optimizing output layer (CRF)
    crf = CRF(tag_type_count)
    out = crf(model)

    # fully specified model
    model = Model(input_, out)

    return model, crf

def compile_model(model_specifications, model, loss_function, metrics):
    """ Compiles a model specified with Keras.

    Args:
        model_specifications: a dictionary containing model hyperparameters.
        model: Keras model object to compile
        loss_function: loss_function to compile model with
        metrics: Keras metrics to compile model with
    """
    # Grab the specs we need to build the model. Keys correspond to names of
    # attributes of SequenceProcessingModel classe.
    learning_rate = model_specifications['learning_rate']
    optimizer = model_specifications['optimizer']

    ## COMPILE
    if optimizer == 'sgd':
        optimizer_ = optimizers.SGD(lr=learning_rate)
    # It is recommended to leave the parameters of this optimizer at their
    # default values (except the learning rate, which can be freely tuned).
    # This optimizer is usually a good choice for recurrent neural networks
    elif optimizer == 'RMSprop':
        optimizer_ = optimizers.RMSprop(lr=learning_rate)

    model.compile(optimizer=optimizer_, loss=loss_function, metrics=[metrics])
    model.summary()
