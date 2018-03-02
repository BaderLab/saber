from keras import optimizers

""" A collection of model helper/utility functions. """

def compile_model(learning_rate, model, optimizer, loss_function, metrics=None):
    """ Compiles a model specified with Keras.

    Args:
        model_specifications: a dictionary containing model hyperparameters.
        model: Keras model object to compile
        loss_function: loss_function to compile model with
        metrics: Keras metrics to compile model with
    """
    ## COMPILE
    if optimizer == 'sgd':
        optimizer_ = optimizers.SGD(lr=learning_rate)
    # It is recommended to leave the parameters of this optimizer at their
    # default values (except the learning rate, which can be freely tuned).
    # This optimizer is usually a good choice for recurrent neural networks
    elif optimizer == 'RMSprop':
        optimizer_ = optimizers.RMSprop(lr=learning_rate)

    model.compile(optimizer=optimizer_, loss=loss_function, metrics=metrics)
    model.summary()
