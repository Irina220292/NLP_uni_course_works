import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

def main(argv):
    """ This script trains a relation prediction model, and outputs predictions for development data.
    It takes three filenames as arguments:

    python3 ./hw10_relation_predictor/baseline_feature_relation_predictor.py <train_file> <dev_file> <prediction_for_dev_file>

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

    # Bring data into matrix format.
    train_tokens, train_labels = augmented_tokens_and_labels(train_file)
    dev_tokens, dev_labels = augmented_tokens_and_labels(dev_file)
    vec = CountVectorizer(tokenizer=lambda x: x.split(' '), ngram_range=(2,3), min_df=2)
    vec.fit(train_tokens)
    le = LabelEncoder()
    le.fit(train_labels)
    train_matrix = vec.transform(train_tokens)
    train_labels = le.transform(train_labels)
    dev_matrix = vec.transform(dev_tokens)

    # Learn classifier and predict answers.
    classifier = linear_model.LogisticRegression()
    classifier.fit(train_matrix, train_labels)
    y_predicted_dev = classifier.predict(dev_matrix)

    # Transform predictions to strings and write out.
    predicted_relation_names = le.inverse_transform(y_predicted_dev)
    with open(prediction_for_dev_file, "w") as text_file:
        print("\n".join(predicted_relation_names), file=text_file)

def augmented_tokens_and_labels(filename):
    """ This reads a file with relation/sentence instances (in tab-separated format) and returns two lists:
    textlist: list of strings; The white-space separated tokens, where each token has a suffix, indicating whether the
        relational subject comes before the object (token_<) or after the object (token_>)
    labellist: list of strings; The relation names for all instances
    """
    textlist = []
    labellist = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            subject_name, relation, object_name, tokens = line.strip().split("\t")
            subject_direction = "_<"
            if tokens.find("<OBJ>") < tokens.find("<SUBJ>"):
                subject_direction = "_>"
            textlist.append(" ".join(token + subject_direction for token in tokens.split(" ")))
            labellist.append(relation)
    return textlist, labellist

if __name__ == "__main__":
    main(sys.argv[1:])
