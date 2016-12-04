
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint


class memn2n(object):

    def __init__(self, vocab_size, story_maxlen, query_maxlen):
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
        self.match.add(Merge([self.input_encoder_m, self.question_encoder], mode='dot', dot_axes=[2, 2]))
        self.match.add(Activation('softmax'))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the input into a single vector with size = story_maxlen:
        self.input_encoder_c = Sequential()
        self.input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen, input_length=story_maxlen))
        self.input_encoder_c.add(Dropout(0.3))
        # output: (samples, story_maxlen, query_maxlen)

        # sum the match vector with the input vector:
        self.response = Sequential()
        self.response.add(Merge([self.match, self.input_encoder_c], mode='sum'))
        # output: (samples, story_maxlen, query_maxlen)
        self.response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)

        # concatenate the match vector with the question vector,
        # and do logistic regression on top
        self.answer = Sequential()
        self.answer.add(Merge([self.response, self.question_encoder], mode='concat', concat_axis=-1))
        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        self.answer.add(LSTM(32))
        # one regularization layer -- more would probably be needed.
        self.answer.add(Dropout(0.3))
        self.answer.add(Dense(vocab_size))
        # we output a probability distribution over the vocabulary
        self.answer.add(Activation('softmax'))

        self.answer.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


    def initialize_checkpoints(self):
        # checkpoint
        self.filepath = "weights.best.hdf5"
        self.checkpoint = ModelCheckpoint(self.filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [self.checkpoint]

    def fit(self, inputs_train, queries_train, answers_train):
        self.answer.fit([inputs_train, queries_train, inputs_train], answers_train, batch_size=32, nb_epoch=120, validation_split=0.1)

    def predict(self, inputs_test, queries_test, answers_test):
        #return self.answer.predict([inputs_test, queries_test, inputs_test], batch_size=32, verbose=1)
        return self.answer.test_on_batch([inputs_test, queries_test, inputs_test], answers_test)