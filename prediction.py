import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import date, timedelta

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close'].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

def train_prophet_model(data):
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    return model

def make_future_predictions(model, periods):
    future_dates = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future_dates)
    return forecast

def plot_results(data, forecast):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['ds'], data['y'], label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='red')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price Prediction')
    ax.legend()
    plt.show()

def main():
    # Set parameters
    ticker = 'AAPL'  # Apple Inc. stock
    start_date = '2020-01-01'
    end_date = date.today().strftime('%Y-%m-%d')
    future_days = 30

    # Fetch stock data
    data = fetch_stock_data(ticker, start_date, end_date)

    # Train Prophet model
    model = train_prophet_model(data)

    # Make future predictions
    forecast = make_future_predictions(model, future_days)

    # Plot results
    plot_results(data, forecast)

    # Print future predictions
    future_predictions = forecast[forecast['ds'] > end_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    print(future_predictions)

if __name__ == "__main__":
    main()