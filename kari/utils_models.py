from keras import optimizers

""" A collection of model helper/utility functions. """

def compile_model(model, learning_rate, optimizer, loss_function):
    """ Compiles a model specified with Keras.

    Args:
        model: Keras model object to compile
        learning_rate (float): learning rate to use during training
        optimizer (str): the optimizer to use during training
        loss_function: Keras loss_function object to compile model with
    """
    if optimizer == 'sgd':
        optimizer_ = optimizers.SGD(lr=learning_rate)
    # It is recommended to leave the parameters of this optimizer at their
    # default values (except the learning rate, which can be freely tuned).
    # This optimizer is usually a good choice for recurrent neural networks
    elif optimizer == 'RMSprop':
        optimizer_ = optimizers.RMSprop(lr=learning_rate)

    model.compile(optimizer=optimizer_, loss=loss_function)
    model.summary()
