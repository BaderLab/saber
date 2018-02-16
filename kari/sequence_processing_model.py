# TODO (johngiorgi): set max_seq_len based on empirical observations

class SequenceProcessingModel(object):
    def __init__(self, num_word_max_seq_len=75):
        self.max_seq_len = max_seq_len

    def _specify(self, num_word_types, num_tag_types, max_seq_len):
        """
        """
        # build the model
        input_ = Input(shape=(max_len,))
        # plus 1 because of '0' word.
        model = Embedding(input_dim=n_words + 1, output_dim=20,
                          input_length=self.max_seq_len, mask_zero=True)(input_)
        model = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.1))(model)
        model = TimeDistributed(Dense(50, activation='relu'))(model)

        crf = CRF(n_tags)
        out = crf(model)

        model = Model(input, out)

        return model, crf


    def _load_parameters(self, parameters_filepath):
        """ Load parameters from ini file if specificed.

        Loads parameters from the ini file if specified, take into account any
        command line argument, and ensure that each parameter is cast to the
        correct type. Command line arguments take precedence over parameters
        specified in the parameter file.

        Args:
            parameters_filepath: path to ini file containing the parameters
        """
