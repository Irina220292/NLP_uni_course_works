from nltk import word_tokenize
from collections import defaultdict
from scipy.sparse import lil_matrix
import math


def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]


def cooccurrences(text, n):
    """
    This takes a string and returns a dictionary mapping tuples of words (word, context_word) to their co-occurrence
    count in windows of n tokens (i.e. the maximum considered distance is n).
    In other words, for each position in the corpus, co-occurrences with n tokens to the left and to the right are
    counted.
    (Note: cooccurrence only holds between words in different positions, not for a position with itself.)

    >>> cooccurrences("a rose is a rose", 2) == {('rose', 'is'): 2, ('rose', 'a'): 3, ('a', 'rose'): 3, ('a', 'is'): 2, \
    ('is', 'rose'): 2, ('is', 'a'): 2}
    True
    >>> cooccurrences("A rose is a rose.", 1) == {('rose', 'is'): 1, ('rose', 'a'): 2, ('a', 'rose'): 2, ('is', 'rose'): 1, ('a', 'is'): 1, ('.', 'rose'): 1, ('rose', '.'): 1, ('is', 'a'): 1}
    True
    >>> cooccurrences("John loves Mary", 5) == {('mary', 'loves'): 1, ('john', 'mary'): 1, ('mary', 'john'): 1, ('loves', 'john'): 1, ('loves', 'mary'): 1, ('john', 'loves'): 1}
    True
    """
    tokens = normalized_tokens(text)
    pair_to_count = defaultdict(int)
    for middle_position in range(len(tokens)):
        middle_word = tokens[middle_position]
        context_start = max(0, middle_position - n)
        context_end = min(len(tokens), middle_position + n + 1)
        for context_position in range(context_start, context_end):
            if context_position != middle_position:
                pair_to_count[(middle_word, tokens[context_position])] += 1
    return pair_to_count


def cooc_dict_to_matrix(cooc_dict):
    """
    This takes a dictionary of word tuples -> counts and returns a dictionary mapping each word to an index, as well as
    a Scipy Sparse matrix containing the counts at the index positions.
    >>> d = {('rose', 'is'): 2, ('rose', 'a'): 3, ('a', 'rose'): 3, ('a', 'is'): 2, ('is', 'rose'): 2, ('is', 'a'): 2}
    >>> m, w2id = cooc_dict_to_matrix(d)
    >>> w2id == {'is': 1, 'a': 0, 'rose': 2}
    True
    >>> m.toarray()
    array([[ 0.,  2.,  3.],
           [ 2.,  0.,  2.],
           [ 3.,  2.,  0.]])
    >>> m.nnz
    6
    """
    vocab = set()
    for w1, w2 in cooc_dict:
        vocab.add(w1)
        vocab.add(w2)
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
    sum_in_col = cooc_matrix.sum(0)  # sparse 1 x d matrix, use sum_in_col[0,i] to get i'th value.
    sum_in_row = cooc_matrix.sum(1)  # sparse d x 1 matrix, use sum_in_row[i,0] to get i'th value.
    ppmi_matrix = lil_matrix(cooc_matrix.shape)
    rows, cols = cooc_matrix.nonzero()
    for row, col in zip(rows, cols):
        sum = cooc_matrix[row, col] / sum_total
        prod = (sum_in_row[row, 0] / sum_total) * (sum_in_col[0, col] / sum_total)
        chance = math.log(sum / prod)
        ppmi_matrix[row, col] = chance if chance > 0 else 0
    return ppmi_matrix
