import random
from nltk import word_tokenize, FreqDist


def dot(dictA, dictB):
    return sum([dictA.get(tok) * dictB.get(tok, 0) for tok in dictA])


def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]


class DataInstance:

    def __init__(self, feature_counts, label):
        """ A data instance consists of a dictionary with feature counts (string -> int) and a label (True or False). """
        self.feature_counts = feature_counts
        self.label = label

    @classmethod
    def from_list_of_feature_occurrences(cls, feature_list, label):
        """ Creates feature counts for all features in the list. """
        feature_counts = dict(FreqDist(feature_list))
        return cls(feature_counts, label)

    @classmethod
    def from_text_file(cls, filename, label):
        with open(filename, 'r') as myfile:
            token_list = normalized_tokens(myfile.read().strip())
        return cls.from_list_of_feature_occurrences(token_list, label)


class Dataset:

    def __init__(self, instance_list):
        """ A data set is defined by a list of instances """
        self.instance_list = instance_list
        self.feature_set = set.union(*[set(inst.feature_counts.keys()) for inst in instance_list])

    def get_topn_features(self, n):
        """ This returns a set with the n most frequently occurring features (i.e. the features that are contained in most instances). """
        top_features = []
        for feat in self.feature_set:
            dist = 0
            for instance in self.instance_list:
                if feat in instance.feature_counts:
                    dist += 1
            top_features.append((feat, dist))
        return set([feat for (feat, dist) in sorted(top_features, key=lambda x: x[1], reverse=True)][:n])

    def set_feature_set(self, feature_set):
        """
        This restrics the feature set. Only features in the specified set all retained. All other feature are removed
        from all instances in the dataset AND from the feature set.
        """
        toDelete = [feat for feat in self.feature_set if feat not in feature_set]
        for feature in toDelete:
            self.feature_set.remove(feature)
            for instance in self.instance_list:
                if feature in instance.feature_counts:
                    instance.feature_counts.pop(feature)

    def most_frequent_sense_accuracy(self):
        """ Computes the accuracy of always predicting the overall most frequent sense for all instances in the dataset. """
        pos, neg = 0, 0
        for instance in self.instance_list:
            if instance.label:
                pos += 1
            else:
                neg += 1
        
        most_frequent_label = pos > neg
        
        correct = 0
        for instance in self.instance_list:
            if instance.label == most_frequent_label:
                correct += 1
        
        return correct / len(self.instance_list)

    def shuffle(self):
        """ Shuffles the dataset. Beneficial for some learning algorithms. """
        random.shuffle(self.instance_list)