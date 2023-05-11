# API Unit Test
# Intégration continue :
# - Test unitaire 2 : fct_extract_X_from_dict(dict_feats)

import numpy as np
import unittest
from api import fct_extract_X_from_dict # module personnel api.py

class TestExtractXFromDict(unittest.TestCase):
    def test_Extract_X_Int(self):
        """
        cas de test 3 : Transformation du dictionnaire des Features de type INTEGER {K, v=integer} en Numpy array X = [[v1, v2,...,vn]] 
        de dimension (1, n), pour l'étape suivante de prédiction du score
        """
        input_data = {'feat_1':'1', 'feat2':'2', 'feat3':'3'}
        result = fct_extract_X_from_dict(input_data)
        expected = np.array([['1', '2', '3']])
        #self.assertEqual(result, expected)
        self.assertIsNone(np.testing.assert_array_equal(result, expected))

    def test_Extract_X_Float(self):
        """
        cas de test 4 : Transformation du dictionnaire des Features de type FLOAT en Numpy array X 
        """
        input_data = {'feat_1':'1.23', 'feat2':'2.34', 'feat3':'3.45'}
        result = fct_extract_X_from_dict(input_data)
        expected = np.array([['1.23', '2.34', '3.45']])
        #self.assertEqual(result, expected)
        self.assertIsNone(np.testing.assert_array_equal(result, expected))

    def test_Extract_X_Nan(self):
        """
        cas de test 5 : Transformation du dictionnaire des Features de tout type INT, FLOAT, Nan en Numpy array X 
        """
        input_data = {'feat_1':'1', 'feat2':'2.34', 'feat3':'np.nan'}
        result = fct_extract_X_from_dict(input_data)
        expected = np.array([['1', '2.34', 'np.nan']])
        #self.assertEqual(result, expected)
        self.assertIsNone(np.testing.assert_array_equal(result, expected))
    
# Le Test Runner appélé depuis la ligne de commande
if __name__ == '__main__':
    unittest.main()