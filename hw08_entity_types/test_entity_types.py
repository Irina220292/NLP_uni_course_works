from unittest import TestCase
from hw08_entity_types import utils
from hw08_entity_types.predict_types import train_evaluate_type_prediction
import numpy as np

class TestEntitiesTypes(TestCase):

    # Exercise 1
    def test_01_read_word_vectors(self):
        """ Tests if word vectors are read correctly."""
        vec_file = "hw08_entity_types/unittest-toydata/word_vectors.txt"
        m, w2id = utils.read_word2vec_file(vec_file)
        self.assertEqual(w2id, {'the': 0, 'on': 1, 'cat': 2, 'sat': 3, 'mat': 4})
        m_expected = np.array([[1, 2, 3],
                               [0.1, 0.2, 0.3],
                               [-1, -2, -3],
                               [0, 0, 0],
                               [-0.1, -0.1, -0.1]])
        self.assertTrue((m == m_expected).all())

    # Exercise 2
    def test_02_read_entity_types_file(self):
        """ Tests if train files are read correctly."""
        vec_file = "hw08_entity_types/unittest-toydata/word_vectors.txt"
        entities_file = "hw08_entity_types/unittest-toydata/entity_types.txt"
        m, w2id = utils.read_word2vec_file(vec_file)
        if m is None:
            assert False, 'Solve exercise 1 first'
        else:
            x, y, type_to_id = utils.read_entity_types_file(entities_file, m, w2id)

            self.assertEqual(set(type_to_id.keys()), {'location', 'item', 'animal'})
            self.assertEqual(set(type_to_id.values()), {0, 1, 2})
            self.assertEqual(x.shape, (5, 3))
            self.assertEqual(y.shape, (5, 3))
            self.assertTrue((x[1] == x[2]).all())
            self.assertEqual((y[1] != y[2]).nnz, 3)
            self.assertEqual(y.sum(), 6)
            self.assertTrue((x[0] == [0.55, 1.1, 1.65]).all())
            self.assertTrue((x[4] == [0, 0, 0]).all())

    # Exercise 3
    def test_03_predict_evaluate(self):
        """ Tests if classification and F1 score are computed correctly."""
        vec_file = "hw08_entity_types/unittest-toydata/word_vectors.txt"
        train_file = "hw08_entity_types/unittest-toydata/entity_types.txt"
        test_file = "hw08_entity_types/unittest-toydata/entity_types.txt"
        fail_file = "hw08_entity_types/unittest-toydata/fail.txt"

        m, w2id = utils.read_word2vec_file(vec_file)
        if m is None:
            assert False, 'Solve exercise 1+2 first'
        else:
            prec, rec, f_score = train_evaluate_type_prediction(vec_file, train_file, test_file)
            self.assertAlmostEqual(prec, 0.6, delta=0.01)
            self.assertAlmostEqual(rec, 0.5, delta=0.01)
            self.assertAlmostEqual(f_score, 0.54, delta=0.01)

            prec, rec, f_score = train_evaluate_type_prediction(vec_file, train_file, fail_file)
            self.assertEqual((prec, rec, f_score), (0, 0, 0))
