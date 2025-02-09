import unittest
from flask import Flask
from flask.testing import FlaskClient
from auto_trading_flask import app

class TestAutoTradingFlask(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Stock Price Predictor', response.data)

    def test_predict_price_no_ticker(self):
        response = self.app.post('/', data=dict(ticker=''))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Please enter a ticker symbol', response.data)

    def test_predict_price_invalid_ticker(self):
        response = self.app.post('/', data=dict(ticker='INVALID'))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'No data found for the ticker symbol', response.data)

if __name__ == '__main__':
    unittest.main()
