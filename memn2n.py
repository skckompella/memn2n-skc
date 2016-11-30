import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

def vectorize_tex():
    pass


class memn2n(object):

    def __init__(self,vocab_size, story_maxlen, query_maxlen):
        # embed the input sequence into a sequence of vectors
        self.input_encoder_m = Sequential()
        self.input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=story_maxlen))
        self.input_encoder_m.add(Dropout(0.3))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the question into a sequence of vectors
        self.question_encoder = Sequential()
        self.question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen))
        self.question_encoder.add(Dropout(0.3))
        # output: (samples, query_maxlen, embedding_dim)

        # compute a 'match' between input sequence elements (which are vectors)
        # and the question vector sequence
        self.match = Sequential()
        self.match.add(Merge([input_encoder_m, question_encoder], mode='dot', dot_axes=[2, 2]))
        self.match.add(Activation('softmax'))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the input into a single vector with size = story_maxlen:
        self.input_encoder_c = Sequential()
        self.input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen, input_length=story_maxlen))
        self.input_encoder_c.add(Dropout(0.3))
        # output: (samples, story_maxlen, query_maxlen)

        # sum the match vector with the input vector:
        self.response = Sequential()
        self.response.add(Merge([match, input_encoder_c], mode='sum'))
        # output: (samples, story_maxlen, query_maxlen)
        self.response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)

        # concatenate the match vector with the question vector,
        # and do logistic regression on top
        self.answer = Sequential()
        self.answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))
        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        self.answer.add(LSTM(32))
        # one regularization layer -- more would probably be needed.
        self.answer.add(Dropout(0.3))
        self.answer.add(Dense(vocab_size))
        # we output a probability distribution over the vocabulary
        self.answer.add(Activation('softmax'))

        self.answer.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        def fit(self, train, test):
            # Note: you could use a Graph model to avoid repeat the input twice
            self.answer.fit([train['inputs'], train['queries'], train['inputs']], train['answers'], batch_size=32, nb_epoch=120,
                                                                validation_data=([test['inputs'], test['queries'], test['inputs']], test['answers']))






