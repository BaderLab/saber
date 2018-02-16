from dataset import Dataset

import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split

# TODO (johngiorgi): set max_seq_len based on empirical observations
# TODO (johngiorgi): consider smarter default values for paramas

class SequenceProcessingModel(object):
    def __init__(self, config_filepath, dataset_text_folder, debug, dropout_rate,
                 gradient_clipping_value, learning_rate, maximum_number_of_epochs,
                 optimizer, output_folder, train_model, max_seq_len=75):
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

        # load dataset
        ds = Dataset(self.dataset_text_folder, max_seq_len=self.max_seq_len)
        ds.load_dataset()
        self.ds = ds

        # split data set into train/test partitions
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            ds.word_idx_sequence, ds.tag_idx_sequence, test_size=0.1)

        # specifiy
        self.model, self.crf = self._specify(ds.word_type_count, ds.tag_type_count)

    def _specify(self, word_type_count, tag_type_count):
        """
        """
        # build the model
        input_ = Input(shape=(self.max_seq_len,))
        # plus 1 because of '0' word.
        model = Embedding(input_dim=word_type_count + 1, output_dim=20,
                          input_length=self.max_seq_len, mask_zero=True)(input_)
        model = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.1))(model)
        model = TimeDistributed(Dense(50, activation='relu'))(model)

        crf = CRF(tag_type_count)
        out = crf(model)

        model = Model(input_, out)

        return model, crf

    def _compile(self):
        if self.optimizer == 'sgd':
            optimizer_ = optimizers.SGD(lr=0.01)

        self.model.compile(optimizer=optimizer_, loss=self.crf.loss_function,
            metrics=[self.crf.accuracy])
        self.model.summary()

    def fit(self):
        """
        """
        # compile
        self._compile()
        # fit
        train_hist = self.model.fit(self.X_train, np.array(self.y_train),
                                    batch_size=32, epochs=6,
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
