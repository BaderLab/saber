from keras import optimizers

"""
A collection of model helper/utility functions.
See https://keras.io/optimizers/ for more info on each optmizer.
"""

# TODO (johngiorgi) add verbosity parameter for printing model summary

def compile_model(model,
                  loss_function,
                  lr=0.01,
                  decay=0.0,
                  optimizer='sgd',
                  verbose=False):
    """Compiles a model specified with Keras.

    Args:
        model: Keras model object to compile
        learning_rate (float): learning rate to use during training
        optimizer (str): the optimizer to use during training
        loss_function: Keras loss_function object to compile model with
    """
    if optimizer == 'sgd':
        optimizer_ = optimizers.SGD(lr=lr, decay=decay)
    # It is recommended to leave the parameters of this optimizer at their
    # default values (except the learning rate, which can be freely tuned).
    # This optimizer is usually a good choice for recurrent neural networks
    elif optimizer == 'rmrprop':
        optimizer_ = optimizers.RMSprop(lr=lr)
    # It is recommended to leave the parameters of this optimizer at their
    # default values.
    elif optimzer == 'adagrad':
        optimizer_ = optimizers.Adagrad()
    # It is recommended to leave the parameters of this optimizer at their
    # default values.
    elif optimzer == 'adadelta':
        optimizer_ = optimizers.Adadelta()
    elif optimizer == 'adam':
        optimizer_ = optimizers.Adam(lr=lr, decay=decay)
    elif optimizer == 'adamax':
        optimizer_ = optimizers.Adamax(lr=lr, decay=decay)
    # It is recommended to leave the parameters of this optimizer at their
    # default values.
    elif optimizer == 'nadam':
        optimizer_ = optimizers.Nadam()

    model.compile(optimizer=optimizer_, loss=loss_function)
    if verbose: model.summary()
