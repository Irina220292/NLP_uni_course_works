from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D
import sys

from hw09_neural_networks import get_data

VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10


def build_and_evaluate_model(x_train, y_train, x_dev, y_dev):
    """ Builds, trains and evaluates a keras LSTM model. """
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_dev = sequence.pad_sequences(x_dev, maxlen=MAX_LEN)
    model = Sequential()
    
    # Add layers.
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=50))
    model.add(Bidirectional(LSTM(units=25)))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Fit to data.
    model.fit(x=x_train, y=y_train, validation_data=(x_dev, y_dev), batch_size=BATCH_SIZE, epochs=EPOCHS)

    score, acc = model.evaluate(x_dev, y_dev)
    return score, acc, model


def main(argv):
    print('Loading data...')
    x_train, y_train, x_dev, y_dev, word2id = get_data.nltk_data(vocab_size=VOCAB_SIZE)
    print(len(x_train), 'training samples')
    print(len(x_dev), 'development samples')
    score, acc, _ = build_and_evaluate_model(x_train, y_train, x_dev, y_dev)
    print('\ndev score:', score)
    print('dev accuracy:', acc)


if __name__ == "__main__":
    main(sys.argv[1:])