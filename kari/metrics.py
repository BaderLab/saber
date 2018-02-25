import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support

# https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
# https://keras.io/metrics/
# https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/

# TODO (johngiorgi): clean up the print function
# TODO (johngiorgi): this is likely copying big lists, find a way to get around
# this

class Metrics(Callback):
    def __init__(self, X_train, y_train, tag_type_to_index):
        self.X_train = X_train
        self.y_train = y_train
        self.tag_type_to_index = tag_type_to_index

    def on_train_begin(self, logs={}):
        """
        """
        ## TRAIN
        self.train_precision_per_epoch = []
        self.train_recall_per_epoch = []
        self.train_f1_per_epoch = []
        ## VALID
        self.valid_precision_per_epoch = []
        self.valid_recall_per_epoch = []
        self.valid_f1_per_epoch = []

    def on_epoch_end(self, epoch, logs={}):
        """
        """
        ## TRAIN
        # get predictions and gold labels
        y_true_train, y_pred_train = self._get_true_and_pred(self.X_train, self.y_train)
        # compute performance metrics
        train_scores = self._get_train_scores(y_true_train, y_pred_train)
        # pretty print a table of performance metrics
        self._pretty_print_train_scores(train_scores)

        ## VALID

    def _get_true_and_pred(self, X, y):
        """ Get y_true and y_pred for given input data (X) and labels (y)

        Performs prediction for the current model (self.model), and returns
        a 2-tuple contain 1D array-like objects containing the true (gold)
        labels and the predicted labels, where labels are integers corresponding
        to the sequence tags as per self.tag_type_to_index.

        Args:
            X: input data
            y: output data

        Returns:
            y_true: 1D array like object containing the gold label sequence
            y_pred: 1D array like object containing the predicted sequences
        """
        # gold labels
        y_true = y.argmax(axis=-1)
        y_true = np.asarray(y_true).ravel()
        # predicted labels
        y_pred = self.model.predict(X).argmax(axis=-1)
        y_pred = np.asarray(y_pred).ravel()

        return y_true, y_pred

    def _get_train_scores(self, y_true, y_pred):
        """ Compute precision, recall, F1 and support for given data.

        For given gold (y_true) and predicted (y_pred) labels, computes
        the precision, recall, F1 and support for all classes (as per
        self.tag_type_to_index)

        Args:
            y_true: 1D array of gold sequence labels
            y_pred: 1D array of predicted sequence labels

        Returns:
            a tuple, where each element is an array representing precision,
            recall, F1 and support scores for a class in self.tag_type_to_index
        """
        # get list of labels,
        labels_ = list(self.tag_type_to_index.values())
        # necc to tell sklearn to supress warnings for 0 scores
        supress = ('precision', 'recall', 'fbeta_score', 'support')

        return precision_recall_fscore_support(y_true, y_pred,
                                               # necc to preserve order
                                               labels = labels_,
                                               # get per label scores
                                               average=None,
                                               warn_for=supress)

    def _pretty_print_train_scores(self, train_scores):
        """ Prints a table of performance scores.

        Given the output of a call to
        sklearn.metrics.precision_recall_fscore_support, prints a table of the
        performance metrics, for each class in self.tag_type_to_index

        Args:
            train_scores: output of a call to
                          sklearn.metrics.precision_recall_fscore_support for
                          one or more classes
        """
        # collect table dimensions
        col_width = 8
        dist_to_first_col = 13
        dist_to_sec_col = 31
        dist_to_third_col = 46
        dist_to_fourth_col = 57
        col_space = ' ' * col_width
        tab_width = 70
        light_line = '-' * tab_width
        heavy_line = '=' * tab_width
        # print header
        print()
        print()
        header = '{l}\nLabel{s}Precision{s}Recall{s}F1{s}Support\n{h}'.format(
            s=col_space, l=light_line, h=heavy_line)
        print(header)
        # print table
        for k, v in self.tag_type_to_index.items():
            print(k, end=' ' * (dist_to_first_col - len(k)))
            print('{0:.2f}'.format(train_scores[0][v]), end=' ' * (dist_to_first_col - 4))
            print('{0:.2f}'.format(train_scores[1][v]), end=' ' * (dist_to_first_col - 4))
            print('{0:.2f}'.format(train_scores[2][v]), end=' ' * (dist_to_first_col - 4))
            print('{}'.format(int(train_scores[3][v])))
        print(heavy_line)
