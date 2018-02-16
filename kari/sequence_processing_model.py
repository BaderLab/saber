import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from dataset import Dataset
from specify_model import specify_LSTM_CRF_
from specify_model import compile_LSTM_CRF_

# TODO (johngiorgi): set max_seq_len based on empirical observations
# TODO (johngiorgi): consider smarter default values for paramas
# TODO (johngiorgi): make sure this process is shuffling the data

class SequenceProcessingModel(object):
    PARAM_DEFAULT = 'default_value_please_ignore_1YVBF48GBG98BGB8432G4874BF74BB'

    def __init__(self,
                 activation_function = PARAM_DEFAULT,
                 batch_size=PARAM_DEFAULT,
                 config_filepath=PARAM_DEFAULT,
                 dataset_text_folder=PARAM_DEFAULT,
                 debug=PARAM_DEFAULT,
                 dropout_rate=PARAM_DEFAULT,
                 gradient_clipping_value=PARAM_DEFAULT,
                 learning_rate=PARAM_DEFAULT,
                 maximum_number_of_epochs=PARAM_DEFAULT,
                 optimizer=PARAM_DEFAULT,
                 output_folder=PARAM_DEFAULT,
                 train_model=PARAM_DEFAULT,
                 max_seq_len=PARAM_DEFAULT
                 ):

        self.activation_function = activation_function
        self.batch_size = batch_size
        self.config_filepath = config_filepath
        self.dataset_text_folder = dataset_text_folder
        self.debug = debug
        self.dropout_rate = dropout_rate
        self.gradient_clipping_value = gradient_clipping_value
        self.learning_rate = learning_rate
        self.maximum_number_of_epochs = maximum_number_of_epochs
        self.optimizer = optimizer
        self.output_folder = output_folder
        self.train_model = train_model
        self.max_seq_len = max_seq_len

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.crf = None
        self.ds = None

        # DATA
        # load dataset
        self.ds = Dataset(self.dataset_text_folder, max_seq_len=self.max_seq_len)
        self.ds.load_dataset()
        # split data set into train/test partitions
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.ds.word_idx_sequence, self.ds.tag_idx_sequence, test_size=0.1)
        # SPECIFY
        # pass a dictionary of this the dataset and model objects attributes as
        # argument to specify_
        self.model, self.crf = specify_LSTM_CRF_({**vars(self.ds), **vars(self)})
        # COMPILE
        compile_LSTM_CRF_(vars(self), self.model, self.crf)

    def fit(self):
        """
        """
        # fit
        train_hist = self.model.fit(self.X_train, np.array(self.y_train),
                                    batch_size=self.batch_size, epochs=6,
                                    validation_split=0.1, verbose=1)

        return pd.DataFrame(train_hist.history)

        '''
        import matplotlib.pyplot as plt
        plt.style.use("ggplot")
        plt.figure(figsize=(12,12))
        plt.plot(hist["acc"])
        plt.plot(hist["val_acc"])
        plt.show()
        '''

    def predict(self):
        p = self.model.predict(np.array([self.X_test[10]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(self.y_test[10], -1)

        print('\n---- Result of BiLSTM-CRF ----\n')

        for w, t, pred in zip(self.X_test[10], true, p[0]):
            if w != 0:
                print("{:15}: {:5} {}".format(self.ds.word_types[w-1], self.ds.tag_types[t], self.ds.tag_types[pred]))


        # self.classification_report(test_y_true, test_y_pred, self.tag_types)

    '''
    def classification_report(y_true, y_pred, labels):
        """Similar to the one in sklearn.metrics, reports per classs recall, precision and F1 score"""
        y_true = numpy.asarray(y_true).ravel()
        y_pred = numpy.asarray(y_pred).ravel()
        corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
        y_true_counts = Counter(y_true)
        y_pred_counts = Counter(y_pred)
        report = ((lab,  # label
                   corrects[i] / max(1, y_true_counts[i]),  # recall
                   corrects[i] / max(1, y_pred_counts[i]),  # precision
                   y_true_counts[i]  # support
                   ) for i, lab in enumerate(labels))
        report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

        print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('', 'recall', 'precision', 'f1-score', 'support'))
        formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
        for r in report:
            print(formatter(*r))
        print('')
        report2 = zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report])
        N = len(y_true)
        print(formatter('avg / total', sum(report2[0]) / N, sum(report2[1]) / N, sum(report2[2]) / N, N) + '\n')
    '''
