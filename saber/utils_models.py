from keras import optimizers

"""
A collection of model helper/utility functions.
"""

# TODO (johngiorgi) add verbosity parameter for printing model summary

def compile_model(model,
                  loss_function,
                  optimizer,
                  lr=0.01,
                  decay=0.0,
                  clipnorm=None,
                  verbose=False):
    """Compiles a model specified with Keras.

    See https://keras.io/optimizers/ for more info on each optmizer.

    Args:
        model: Keras model object to compile
        loss_function: Keras loss_function object to compile model with
        optimizer (str): the optimizer to use during training
        lr (float): learning rate to use during training
        decay (float): per epoch decay rate
        clipnorm (float): gradient normalization threshold
        verbose (bool): if True, prints model summary after compilation
    """
    # The parameters of these optimizers can be freely tuned.
    if optimizer == 'sgd':
        optimizer_ = optimizers.SGD(lr=lr, decay=decay, clipnorm=clipnorm)
    elif optimizer == 'adam':
        optimizer_ = optimizers.Adam(lr=lr, decay=decay, clipnorm=clipnorm)
    elif optimizer == 'adamax':
        optimizer_ = optimizers.Adamax(lr=lr, decay=decay, clipnorm=clipnorm)
    # It is recommended to leave the parameters of this optimizer at their
    # default values (except the learning rate, which can be freely tuned).
    # This optimizer is usually a good choice for recurrent neural networks
    elif optimizer == 'rmrprop':
        optimizer_ = optimizers.RMSprop(lr=lr, clipnorm=clipnorm)
    # It is recommended to leave the parameters of these optimizers at their
    # default values.
    elif optimizer == 'adagrad':
        optimizer_ = optimizers.Adagrad(clipnorm=clipnorm)
    elif optimizer == 'adadelta':
        optimizer_ = optimizers.Adadelta(clipnorm=clipnorm)
    elif optimizer == 'nadam':
        optimizer_ = optimizers.Nadam(clipnorm=clipnorm)

    model.compile(optimizer=optimizer_, loss=loss_function)
    if verbose: model.summary()

def precision_recall_f1_support(true_positives, false_positives, false_negatives):
    """Returns the precision, recall, F1 and support from TP, FP and FN counts.

    Returns a four-tuple containing the precision, recall, F1-score and support
    For the given true_positive (TP), false_positive (FP) and
    false_negative (FN) counts.

    Args:
        true_positives (int): number of true-positives predicted by classifier
        false_positives (int): number of false-positives predicted by classifier
        false_negatives (int): number of false-negatives predicted by classifier

    Returns:
        four-tuple containing (precision, recall, f1, support)
    """
    precision = true_positives / (true_positives + false_positives) \
        if true_positives > 0 else 0.
    recall = true_positives / (true_positives + false_negatives) \
        if true_positives > 0 else 0.
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.
    support = true_positives + false_negatives

    return precision, recall, f1, support
