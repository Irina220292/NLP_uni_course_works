import math, random
import numpy as np
from collections import Counter


def vocabulary_to_id_for_wordlist(word_list, vocab_size):
    """ Returns a mapping from word to id for vocab_size most frequent words in a list. """
    mostcommon = [w for (w, c) in Counter(word_list).most_common(vocab_size)]
    return {w : i for (i, w) in enumerate(mostcommon)}


def sigmoid(x):
    """ Calculates the logistic sigmoid function. """
    return 1 / (1 + math.e ** -x)


def positive_and_negative_cooccurrences(tokens, max_distance, neg_samples_factor, vocab_to_id):
    """
    This takes a list of strings (words) and returns a generator of word_id tuples of the following form:
    (target_word_id, context_word_id, True)
    or
    (target_word_id, random_word_id, False)
    where target_word is the word at a certain position, and context_word is at most "max_distance" tokens away.
    For each observed word pair (with label True), neg_samples_factor word pairs (with label False) are added, where
    the context word is replaced by a random word.

    Words are represented by integers (rather than by string) denoting their id.
    Only pairs where both words are in the vocabulary are considered.

    (Note: cooccurrence only holds between words in different positions, not for a position with itself.)
    :param tokens: list of strings (words)
    :param max_distance: max distance of context word to target word
    :param neg_samples_factor: number of sampled negative tuples for each positive tuple
    :param vocab_to_id: dictionary (string to int) mapping each word to its id (=row in embedding matrizes).
    :return: generator over tuples of the form (target_word_id:int, context_word_id:int, label:boolean)
    """
    for target_position in range(len(tokens)):
        context_start = max(0, target_position - max_distance)
        context_end = min(len(tokens), target_position + max_distance + 1)
        for context_position in range(context_start, context_end):
            if context_position != target_position:
                yield (vocab_to_id[tokens[target_position]], vocab_to_id[tokens[context_position]], True)
                for c in range(neg_samples_factor):
                    yield (vocab_to_id[tokens[target_position]], random.randint(0, len(vocab_to_id) - 1), False)


class DenseSimilarityMatrix:

    def __init__(self, word_matrix, word_to_id):
        """
        Creates a WordSimilarity object.

        :param word_matrix: A matrix-like object (numpy 2d array or scipy sparse matrix), where rows correspond to words
            and columns correspond to dimensions of the representation space (context or embedding feature).
        :param word_to_id: A dictionary from word string to word id (= row number in word_matrix).
        """
        self.word_matrix = word_matrix
        self.word_to_id = word_to_id
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

    def word_similarity(self, wordA, wordB):
        if not (wordA in self.word_to_id and wordB in self.word_to_id):
            return .0
        rowA, rowB = (self.word_to_id[wordA], self.word_to_id[wordB])
        vecA, vecB = (self.word_matrix[rowA, :], self.word_matrix[rowB, :])
        dotAB, dotAA, dotBB = (vecA.dot(vecB.T), vecA.dot(vecA.T), vecB.dot(vecB.T))
        return dotAB / math.sqrt(dotAA * dotBB)

    def most_similar_words(self, word, n):
        if word not in self.word_to_id:
            return []
        row = self.word_to_id[word]
        vec = self.word_matrix[row, :]
        m = self.word_matrix
        dot_m_v = m.dot(vec.T)  # n-dim vector
        dot_m_m = np.sum(m * m, axis=1)  # n-dim vector
        dot_v_v = vec.dot(vec.T)  # float
        sims = dot_m_v / (math.sqrt(dot_v_v) * np.sqrt(dot_m_m))
        return [self.id_to_word[idx] for idx in (-sims).argsort()[:n]]