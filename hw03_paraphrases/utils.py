import random
from collections import Counter
from nltk import word_tokenize

def dot(dictA, dictB):
    return sum([dictA.get(tok) * dictB.get(tok,0) for tok in dictA])

def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]

class DataInstance:
    def __init__(self, feature_counts, label):
        """ A data instance consists of a dictionary with feature counts (string -> int) and a label (True or False)."""
        self.feature_counts = feature_counts
        self.label = label

    @classmethod
    def from_list_of_feature_occurrences(cls, feature_list, label):
        """ Creates feature counts for all features in the list."""
        feature_counts = dict()
        for feature in feature_list:
            count = feature_counts.get(feature, 0)
            feature_counts[feature] = count + 1
        return cls(feature_counts, label)

    @classmethod
    def from_text_file(cls, filename, label):
        with open(filename, 'r', encoding='ISO-8859-1') as myfile:
            token_list = normalized_tokens(myfile.read().strip())
        return cls.from_list_of_feature_occurrences(token_list, label)

class Dataset:
    def __init__(self, instance_list):
        """ A data set is defined by a list of instances """
        self.instance_list = instance_list
        self.feature_set = set.union(*[set(inst.feature_counts.keys()) for inst in instance_list])


    def get_topn_features(self, n):
        """ This returns a set with the n most frequently occurring features (i.e. the features that are contained in most instances)."""
        feat_count = Counter()
        for instance in self.instance_list:
            feat_count.update(instance.feature_counts.keys())
        return {f_c[0] for f_c in feat_count.most_common(n)}

    def set_feature_set(self, feature_set):
        """
        This restrics the feature set. Only features in the specified set all retained. All other feature are removed
        from all instances in the dataset AND from the feature set."""
        self.feature_set = feature_set
        for instance in self.instance_list:
            filtered_feature_count = {f:instance.feature_counts[f] for f in instance.feature_counts
                                      if f in self.feature_set}
            instance.feature_counts = filtered_feature_count

    def most_frequent_sense_accuracy(self):
        """ Computes the accuracy of always predicting the overall most frequent sense for all instances in the dataset. """
        countTrue = countFalse = 0
        for inst in self.instance_list:
            if inst.label == True:
                countTrue += 1
            else:
                countFalse +=1
        return max(countTrue, countFalse) / (countTrue + countFalse)

    def shuffle(self):
        """ Shuffles the dataset. Beneficial for some learning algorithms."""
        random.shuffle(self.instance_list)
