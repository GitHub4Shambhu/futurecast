# Future Forecaster

Future Forecaster (a.k.a Futurecast) is a web application that predicts stock prices using Prophet and LSTM models. Users can enter a ticker symbol and receive interactive visualizations plus tabular views for the next 30 business days.

## Features

- Stock price prediction using Facebook Prophet
- Deep learning predictions with LSTM
- Interactive web interface with charts and tables
- Visual representations of forecasts and error bands
- 30-day business-day forecasting horizon

## Installation

1. Clone the repository:
```sh
git clone <your-repository-url>
cd futurecast
```

2. (Recommended) Create a virtual environment and activate it:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use "venv\\Scripts\\activate"
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:

```sh
python auto_trading_flask.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000/` (or `http://localhost:5000`).

3. Enter a stock ticker symbol (e.g., AAPL, GOOGL) and click **Predict** to see the 30-day forecast, tables, and plots.

## Models

- **Prophet** – captures seasonality and trend for classical time-series forecasting.
- **LSTM** – deep learning model that learns complex temporal patterns for complementary predictions.

## Project Structure

```
futurecast/
├── auto_trading_flask.py
├── auto_trading.py
├── lstm_model.py
├── prediction.py
├── test_auto_trading.py
├── test_auto_trading_flask.py
├── requirements.txt
└── README.md
```

## Tech Stack

- Flask
- Prophet
- TensorFlow/Keras
- yfinance
- pandas
- seaborn
- matplotlib

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
