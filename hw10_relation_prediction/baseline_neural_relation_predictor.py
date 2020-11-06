import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(13370)
tf_session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1))
import sys
from collections import Counter
from keras.models import Sequential
from keras.layers import Embedding, GlobalMaxPooling1D, Dense
from keras import backend as K
K.set_session(tf_session)
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle

hw10_path = "hw10_relation_prediction"


def main(argv):
    """ This script trains a relation prediction model, and outputs predictions for development data.
    It takes three filenames as arguments:

    python3 ./hw10_relation_predictor/baseline_neural_relation_predictor.py <train_file> <dev_file> <prediction_for_dev_file>

    The files <train_file> and <dev_file> have the following tab-separated column format:

    "relational_subject_entity" \t "relation_name" \t "relational_object_entity" \t "sentence"

    The sentence is white-space separated, and subject / object entities are replaced by the dummy tokens <SUBJ> / <OBJ>.
    The script can assume that all relation types occur at least once in the training data.

    The file name <prediction_for_dev_file> indicates where the model prediction for the development data should be
    saved to. The output format is the predicted "relation_name" for each line in <dev_file>.

    If the output format is correct, the following will calculate the accuracy of the model:

    python3 ./hw10_relation_predictor/evaluate.py <dev_file> <prediction_for_dev_file>
    """
    train_file, dev_file, prediction_for_dev_file = argv

    train_tokens, train_labels = tokens_and_labels(train_file)
    dev_tokens, dev_labels = tokens_and_labels(dev_file)

    # Convert text and labels to matrix format.
    vocab_size = 1000
    word_to_id = get_word_to_id_map(train_tokens, vocab_size)
    le = LabelEncoder()
    le.fit(train_labels)
    maxlen=50
    num_classes = 40

    train_matrix = get_text_matrix(train_tokens, word_to_id, maxlen)
    train_labels = to_categorical(le.transform(train_labels), num_classes)
    dev_matrix = get_text_matrix(dev_tokens, word_to_id, maxlen)
    dev_labels = to_categorical(le.transform(dev_labels), num_classes)

    # Training and prediction
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_matrix, train_labels, validation_data=(dev_matrix, dev_labels),epochs=10)
    y_predicted_dev = model.predict_classes(dev_matrix)

    # Transform predictions to strings and write out.
    predicted_relation_names = le.inverse_transform(y_predicted_dev)
    with open(prediction_for_dev_file, "w") as text_file:
        print("\n".join(predicted_relation_names), file=text_file)

def tokens_and_labels(filename):
    """ This reads a file with relation/sentence instances (in tab-sepated format) and returns two lists:
    textlist: list of strings; The white-space seprated tokens.
    labellist: list of strings; The relation names for all instances.
    """
    textlist = []
    labellist = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            subject_name, relation, object_name, tokens = line.strip().split("\t")
            textlist.append(tokens)
            labellist.append(relation)
    return textlist, labellist

def get_word_to_id_map(textlist, vocab_size):
    """ Creates mapping word -> id for the most frequent words in the vocabulary. Ids 0 and 1 are reserved for the
    padding symbol <PAD> and the unknown token <UNK>. vocab_size determines the overall vocabulary size (including <UNK>
    and <PAD>)"""
    assert(vocab_size >= 2)
    c = Counter(tok for text in textlist for tok in text.split(" "))
    try:
        # Use fixed word frequencies
        with open(hw10_path + "/ids_words" + str(vocab_size) + ".p", "rb") as f:
            ids_words = pickle.load(f)
    except FileNotFoundError:
        ids_words = enumerate(['<PAD>','<UNK>'] + sorted([word for word, count in c.most_common(vocab_size - 2)]))
        # Write file instead of extensive deterministic function
        with open(hw10_path + "/ids_words" + str(vocab_size) + ".p", "wb") as f:
            pickle.dump(ids_words, f)
    return {w: idx for idx, w in ids_words}

def get_text_matrix(textlist, word_to_id, maxlen):
    """ This takes textlist (list with white-space separated tokens) and returns a numpy matrix of size
    len(textlist) x maxlen.
    Each row in the matrix contains for each text the sequence of word ids (i.e. the columns correspond to the positions
    in the sentence).
    If a sentence is longer than maxlen, it is truncated after maxlen tokens.
    If a sentence is shorter than maxlen, it is filled with 0 (= the word id of the <PAD> token).
    """
    m = np.zeros((len(textlist), maxlen), dtype=int)
    row_nr, col_nr = 0, 0
    for text in textlist:
        col_nr = 0
        for word in text.split(" "):
            if col_nr == maxlen:
                break
            m[row_nr, col_nr] = word_to_id.get(word, 1) # id for <UNK>
            col_nr += 1
        row_nr += 1
    return m

if __name__ == "__main__":
    main(sys.argv[1:])
