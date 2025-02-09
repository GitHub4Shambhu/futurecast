import unittest
import pandas as pd
import numpy as np
from auto_trading import tickerSymbol, tickerData, tickerDf, features, labels, features_train, features_test, labels_train, labels_test, model, predictions

class TestAutoTrading(unittest.TestCase):
    def test_ticker_data(self):
        # Check if tickerData is not None
        self.assertIsNotNone(tickerData)

    def test_historical_data(self):
        # Check if tickerDf is a DataFrame and not empty
        self.assertIsInstance(tickerDf, pd.DataFrame)
        self.assertFalse(tickerDf.empty)

    def test_features_labels(self):
        # Check if features and labels are not None
        self.assertIsNotNone(features)
        self.assertIsNotNone(labels)

    def test_data_split(self):
        # Check if data is split into training and test sets
        self.assertIsNotNone(features_train)
        self.assertIsNotNone(features_test)
        self.assertIsNotNone(labels_train)
        self.assertIsNotNone(labels_test)

    def test_model_training(self):
        # Check if model is trained (model.coef_ and model.intercept_ are not None)
        self.assertIsNotNone(model.coef_)
        self.assertIsNotNone(model.intercept_)

    def test_predictions(self):
        # Check if predictions are made
        self.assertIsInstance(predictions, np.ndarray)
        self.assertFalse(predictions.size == 0)

if __name__ == '__main__':
    unittest.main()