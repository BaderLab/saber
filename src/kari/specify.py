from keras_contrib.layers import CRF
from keras.models import Model, Input
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

# there is a bunch of preprocessing steps I need to perform.
# 1: I need to append a "sentence number" to each "sentence group" (not entirely sure if this is necc.)
# 2: need to get POS tags (also not necc. but would be good to do)

# get the number of sentences
# with open('train.tsv', 'r') as f:
#    num_sentences = len(f.read().split('\n\n'))

# read in data set, change default delim and add column names
# data = pd.read_csv('train.tsv', sep='\t', encoding='latin1',
#                     names = ["Word", "Tag"])

def specify(n_words, n_tags, max_len=75):
    # build the model
    input = Input(shape=(max_len,))
    # plus 1 because of '0' word.
    model = Embedding(input_dim=n_words + 1, output_dim=20,
                      input_length=max_len, mask_zero=True)(input)
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(50, activation='relu'))(model)

    crf = CRF(n_tags)
    out = crf(model)

    model = Model(input, out)

    return model, crf
