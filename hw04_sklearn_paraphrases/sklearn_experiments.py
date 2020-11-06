from sklearn.feature_extraction import DictVectorizer

list_of_dicts_1 = [{'a':1, 'b':3, 'c':2}, {'a':2, 'c':3}]
list_of_dicts_2 = [{'a':1, 'x':3, 'y':4}, {'a':2, 'b':3}]


def print_sparse_matrix(M):
    """  Prints a sparse matrix in dense, nested format (i.e. also shows elements that are zero). """
    print(M.todense())


def make_matrix1(list_of_dicts):
    """
    This creates a Dictvectorizer from a list of dictionaries, and uses it to create a Scipy sparse matrix
    containing the feature values of the dictionaries. The matrix is returned (the vectorizer is not returned).
    """
    v = DictVectorizer()
    XYZ = v.fit_transform(list_of_dicts)
    return XYZ


def make_matrix2(list_of_dictsA, list_of_dictsB):
    """
    This creates a Dictvectorizer from a list A of dictionaries, and uses it to create a Scipy sparse matrix
    from a different list B of dictionaries (i.e. features that were not in list A are ignored).
    The matrix is returned (the vectorizer is not returned).
    """
    v = DictVectorizer()
    XYZ = v.fit_transform(list_of_dictsA)
    v.inverse_transform(XYZ)
    return v.transform(list_of_dictsB)


# Uncomment for inspecting the results of your implementation.
x = make_matrix1(list_of_dicts_1)
print_sparse_matrix(x)
y = make_matrix2(list_of_dicts_1, list_of_dicts_2)
print_sparse_matrix(y)
