import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import LSTM, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

class RNN:
    '''
    RNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, dict_size=5000, example_length=500, embedding_length=128, epochs=2, batch_size=64):
        '''
        initialize RNN model
        :param train_x: training data
        :param train_y: training label
        :param test_x: test data
        :param test_y: test label
        :param epochs:
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.example_len = example_length
        self.dict_size = dict_size
        self.embedding_len = embedding_length

        # TODO:preprocess training data
        self.train_x = sequence.pad_sequences(train_x, maxlen=self.example_len)
        self.test_x = sequence.pad_sequences(test_x, maxlen=self.example_len)
        self.train_y = train_y
        self.test_y = test_y

        # TODO:build model
        self.model = Sequential()
        self.model.add(Embedding(self.dict_size, self.embedding_len, input_length=self.example_len))
        self.model.add(Dropout(0.25))
        self.model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1)) 
        self.model.add(MaxPooling1D(pool_size=4)) 
        self.model.add(LSTM(128, kernel_initializer='random_uniform', bias_initializer='zeros', dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])


    def train(self):
        '''
        fit in data and train model
        :return:
        '''
        # TODO: fit in data to train your model
        self.model.fit(self.train_x, self.train_y,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(self.test_x, self.test_y))

    def evaluate(self):
        '''
        evaluate trained model
        :return:
        '''
        return self.model.evaluate(self.test_x, self.test_y)


if __name__ == '__main__':
    max_words = 5000
    example_len = 500
    embedding_len = 128
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=max_words)
    rnn = RNN(train_x, train_y, test_x, test_y, max_words, example_len, embedding_len)
    rnn.train()
    score, acc = rnn.evaluate()
    print("score:", score, "accuracy:", acc)

