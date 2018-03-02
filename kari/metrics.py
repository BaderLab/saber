import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support

# https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
# https://keras.io/metrics/
# https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/

# TODO (johngiorgi): there is some hard coded ugliness going on in print_table,
# fix this.
# TODO (johngiorgi): this is likely copying big lists, find a way to get around
# this


# For all keys in train/valid scores, group the values based on shared class
#

class Metrics(Callback):
    def __init__(self, X_train, X_valid, y_train, y_valid, tag_type_to_index):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
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
        train_scores = self._get_performance_scores(y_true_train, y_pred_train)
        # pretty print a table of performance metrics
        self._pretty_print_performance_scores(train_scores)
        # updates per epoch metric accumulators
        self.train_precision_per_epoch.append(train_scores[0])
        self.train_recall_per_epoch.append(train_scores[1])
        self.train_f1_per_epoch.append(train_scores[2])
        ## VALID
        y_true_valid, y_pred_valid = self._get_true_and_pred(self.X_valid, self.y_valid)

        valid_scores = self._get_performance_scores(y_true_valid, y_pred_valid)

        self._pretty_print_performance_scores(valid_scores, title='valid')

        self.valid_precision_per_epoch.append(valid_scores[0])
        self.valid_recall_per_epoch.append(valid_scores[1])
        self.valid_f1_per_epoch.append(valid_scores[2])

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

    def _get_performance_scores(self, y_true, y_pred):
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

    '''
    def _extract_combined_scores(self, performance_scores):
        """
        """
        class_groups = list(set([k.split('-')[0] for k in performance_scores.keys()]))
        for class_ in class_groups:
            for k in performance_scores.keys():
                if class_ == k.split('-')[1]

        # get indices for all labels that are not the 'null' label, 'O'.
        # labels_ = [(k, v)[1] for k, v in self.ds.word_type_to_idx.items() if k != 'O']
    '''


    def _pretty_print_performance_scores(self, performance_scores, title='train'):
        """ Prints a table of performance scores.

        Given the output of a call to
        sklearn.metrics.precision_recall_fscore_support, prints a table of the
        performance metrics, for each class in self.tag_type_to_index

        Args:
            performance_scores: output of a call to
                          sklearn.metrics.precision_recall_fscore_support for
                          one or more classes
        """
        # collect table dimensions
        col_width = 20
        col_space = ' ' * col_width
        col_width_1 = len('Label') + col_width
        col_width_2 = col_width_1 + len('Precision') + col_width
        col_width_3 = col_width_2 + len('Recall') + col_width
        col_width_4 = col_width_3 + len('F1') + col_width

        tab_width = 120
        light_line = '-' * tab_width
        heavy_line = '=' * tab_width

        ## HEADER
        print()
        print()
        title = '{col}{t}{col}'.format(t=title.upper(),
                                       col=' '*((tab_width-len(title))//2))
        header = 'Label{col}Precision{col}Recall{col}F1{col}Support'.format(col=col_space)
        print(heavy_line)
        print(title)
        print(light_line)
        print(header)

        print(heavy_line)
        ## BODY
        for k, v in self.tag_type_to_index.items():
            # specify an entire row
            row = '{lab}{col1}{p:.2f}{col2}{r:.2f}{col3}{f1:.2f}{col4}{s}'.format(
                p=performance_scores[0][v],
                r=performance_scores[1][v],
                f1=performance_scores[2][v],
                s=int(performance_scores[3][v]),
                lab=k,
                col1=' ' * (col_width_1 - len(k) + len('Precision')//3 - 1),
                col2=' ' * (col_width_2 - col_width_1 - len('Precision')//3 - 1),
                col3=' ' * (col_width_3 - col_width_2 - len('Precision')//3 - 1),
                col4=' ' * (col_width_4 - col_width_3 - len('Precision')//3 - 1))
            print(row)
        print(light_line)

        ## FOOTER
        # get average scores for each label
        avg_score_per_label = [np.mean(metric) for metric in performance_scores]
        print('AVERAGE{col1}{0:.2f}{col2}{0:.2f}{col3}{0:.2f}'.format(
            avg_score_per_label[0],
            avg_score_per_label[1],
            avg_score_per_label[2],
            col1=' ' * (col_width_1 - len('AVERAGE') + len('Precision')//3 - 1),
            col2=' ' * (col_width_2 - col_width_1 - len('Precision')//3 - 1),
            col3=' ' * (col_width_3 - col_width_2 - len('Precision')//3 - 1)))
        print(heavy_line)
