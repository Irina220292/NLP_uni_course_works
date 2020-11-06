from collections import defaultdict, Counter
from scipy.sparse import lil_matrix
from functools import lru_cache
import math


def vocabulary_from_wordlist(word_list, vocab_size):
    """ Returns set of vocab_size most frequent words from a given list of words.

    >>> v = vocabulary_from_wordlist(['a','rose', 'is', 'a', 'rose', 'colour', 'rose'],2)
    >>> v == {'a', 'rose'}
    True
    >>> v = vocabulary_from_wordlist(['a','rose', 'is', 'a', 'rose'],4)
    >>> v == {'a', 'rose', 'is'}
    True
    """
    mostcommon = Counter(word_list).most_common(vocab_size)
    return set([w for (w, c) in mostcommon])


def cooccurrences(tokens, n, vocab):
    """
    This takes a list of tokens (representing a text) and returns a dictionary mapping tuples of words
    to their co-occurrence count in windows of n tokens (i.e. the maximum considered distance is n).
    In other words, for each position in the corpus, co-occurrences with n tokens to the left and to the right are
    counted. Only words in a given set of words (the vocabulary) are considered.
    (Note: co-occurrence only holds between words in different positions, not for a position with itself.)

    >>> cooccurrences(["a","rose","is","a","rose"], 2, {"rose", "a"}) == {('rose', 'a'): 3, ('a', 'rose'): 3}
    True
    >>> cooccurrences(["a","rose","is","a","rose"], 1, {"rose", "is"}) == {('rose', 'is'): 1, ('is', 'rose'): 1}
    True
    """
    pair_to_count = defaultdict(int)
    for middle_position in range(len(tokens)):
        middle_word = tokens[middle_position]
        context_start = max(0, middle_position - n)
        context_end = min(len(tokens), middle_position + n + 1)
        for context_position in range(context_start, context_end):
            if context_position != middle_position and middle_word in vocab and tokens[context_position] in vocab:
                pair_to_count[(middle_word, tokens[context_position])] += 1
    return pair_to_count


def cooc_dict_to_matrix(cooc_dict, vocab):
    """
    This takes a dictionary (word tuples/co-occurrences -> counts) and a vocabulary;
    returns a dictionary mapping each word to an index, as well as
    a Scipy Sparse matrix containing the counts at the index positions.
    >>> d = {('rose', 'is'): 2, ('rose', 'a'): 3, ('a', 'rose'): 3, ('a', 'is'): 4, ('is', 'rose'): 5, ('is', 'a'): 6}
    >>> m, w2id = cooc_dict_to_matrix(d, {'a', 'rose', 'is'})
    >>> w2id == {'is': 1, 'a': 0, 'rose': 2}
    True
    >>> m.toarray()
    array([[ 0.,  4.,  3.],
           [ 6.,  0.,  5.],
           [ 3.,  2.,  0.]])
    >>> m.nnz
    6
    """
    word_to_id = {w:i for i, w in enumerate(sorted(vocab))}
    m = lil_matrix((len(vocab), len(vocab)))
    for ((w1, w2), v) in cooc_dict.items():
        m[word_to_id[w1], word_to_id[w2]] += v
    return m, word_to_id


def ppmi_weight(cooc_matrix):
    """
    This computes a PPMI weighted version of a square matrix with non-negative elements, i.e. a new matrix is returned
    that contains for each cell of the original matrix its PPMI.

    The pointwise information is defined as:
    PMI = log( P(r,c) / (P(r)*P(c)) )
    Where r,c stand for rows and columns of the matrix and:
    P(r,c) = value_of_cell_r_c / sum_of_all_cells
    P(r) = value_of_cells_in_row_r / sum_of_all_cells
    P(c) = value_of_cells_in_column_c / sum_of_all_cells

    The PPMI keeps the positive PMI values, and replaces all negative (or undefined) values with 0.

    >>> m = lil_matrix([[1,2],[3,4]])
    >>> ppmi_weight(m).toarray()
    array([[ 0.        ,  0.10536052],
           [ 0.06899287,  0.        ]])
    """
    sum_total = cooc_matrix.sum()
    sum_in_col = cooc_matrix.sum(0)
    sum_in_row = cooc_matrix.sum(1)
    ppmi_matrix = lil_matrix(cooc_matrix.shape)
    rows, cols = cooc_matrix.nonzero()
    for row, col in zip(rows, cols):
        sum = cooc_matrix[row, col] / sum_total
        prod = (sum_in_row[row, 0] / sum_total) * (sum_in_col[0, col] / sum_total)
        chance = math.log(sum / prod)
        ppmi_matrix[row, col] = chance if chance > 0 else 0
    return ppmi_matrix