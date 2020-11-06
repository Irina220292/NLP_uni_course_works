from unittest import TestCase
from hw08_entity_types.predict_types import train_evaluate_type_prediction

class TestEntitiesTypesLarge(TestCase):

    def test_04_predict_evaluate_large(self):
        """ Tests if classification and F1 score are computed correctly."""
        vec_file = "data/entities_types/word_vectors.txt"
        train_file = "data/entities_types/names_types.train"
        test_file = "data/entities_types/names_types.test"

        prec, rec, f_score = train_evaluate_type_prediction(vec_file, train_file, test_file)
        print("\nprec, rec, f_score")
        print(prec, rec, f_score)

        self.assertAlmostEqual(prec, 0.81, delta=0.05)
        self.assertAlmostEqual(rec, 0.48, delta=0.05)
        self.assertAlmostEqual(f_score, 0.6, delta=0.05)