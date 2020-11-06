# -*- coding: utf-8 -*-
import sys


def main(argv):
    """ This script compares the gold labels of a (training, dev or test) file with a model prediction, and prints the
    accuracy to the standard output.

    python3 ./hw10_relation_predictor/evaluate.py <dev_file> <prediction_for_dev_file>
    """
    gold_file, predicted_file = argv
    with open(gold_file, encoding='utf-8') as f:
        gold_labels = [line.split("\t")[1] for line in f]
    with open(predicted_file, encoding='utf-8') as f:
        predicted_labels = [line.strip() for line in f]
    num_correct = sum(gold == predicted for gold, predicted in zip(gold_labels, predicted_labels))
    accuracy = num_correct / len(gold_labels)
    print("Accuracy: \n{0:.4f}".format(accuracy))
    return accuracy

if __name__ == "__main__":
    main(sys.argv[1:])
