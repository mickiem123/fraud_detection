import unittest
import pandas as pd
import os
from features_engineering import Preprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Load a small sample of data for testing
        data_path = os.path.join(os.path.dirname(__file__), '../../data/raw_data/2018-04-01.pkl')
        self.df = pd.read_pickle(data_path)
        self.preprocessor = Preprocessor(self.df)

    def test_get_TERMINAL_ID_characteristic(self):
        # Should not raise and should add expected columns
        self.preprocessor.get_TERMINAL_ID_characteristic()
        for size in self.preprocessor.config.windows_size:
            nb_tx_col = f'TERMINAL_ID_NB_TX_{size}DAY_WINDOW'
            risk_col = f'TERMINAL_ID_RISK_{size}DAY_WINDOW'
            self.assertIn(nb_tx_col, self.preprocessor.df.columns)
            self.assertIn(risk_col, self.preprocessor.df.columns)

if __name__ == "__main__":
    unittest.main()
