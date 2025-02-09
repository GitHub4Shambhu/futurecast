# Future Forecaster

A web application that predicts stock prices using both Prophet and LSTM models.

## Features

- Stock price prediction using Facebook Prophet
- Deep Learning predictions using LSTM
- Interactive web interface
- Visual representations of predictions
- 30-day price forecasting

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd futurecast
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python auto_trading_flask.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter a stock ticker symbol (e.g., AAPL, GOOGL) and click "Predict" to see the forecast.

## Models

- **Prophet**: Used for primary forecasting, incorporating seasonal trends
- **LSTM**: Deep learning model for complex pattern recognition in stock data

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
