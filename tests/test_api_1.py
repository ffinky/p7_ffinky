# API Unit Test
# Intégration continue :
# - Test unitaire 1 : fct_load_classifier(choice_model)

import unittest
from api import fct_load_classifier  # module personnel api.py

# Test unitaire 1 : fct_load_classifier(choice_model)
class TestLoadClassifier(unittest.TestCase):
    def test_load_RegrLogistic(self):
        """
        cas de test 1 : load du modèle de regression logistique SI input = 'Reg Logistique'
        """
        input_data = 'Reg Logistique'
        result = fct_load_classifier(input_data)
        self.assertEqual(result['classifier'].__class__.__name__, 'LogisticRegression') # résultat attendu

    def test_load_LGBM(self):
        """
        cas de test 2 : load du modèle Light GBM SI input != 'Reg Logistique'
        """
        input_data = 'LGBM'
        result = fct_load_classifier(input_data)
        self.assertEqual(result['classifier'].__class__.__name__, 'LGBMClassifier') # résultat attendu

# Le Test Runner appélé depuis la ligne de commande
if __name__ == '__main__':
    unittest.main()