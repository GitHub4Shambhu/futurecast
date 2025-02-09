import pandas as pd
import yfinance as yf
from prophet import Prophet
import streamlit as st

def predict_price(tickerSymbol):
    if not tickerSymbol:
        st.error("Please enter a ticker symbol")
        return

    st.write(f"Ticker Symbol: {tickerSymbol}")

    # Get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='1d', start='2020-1-1', end='2024-12-31')

    if tickerDf.empty:
        st.error("No data found for the ticker symbol")
        return

    st.write("Data fetched successfully")

    # Prepare data for Prophet
    df = tickerDf.reset_index()[['Date', 'Close']]
    df['Date'] = df['Date'].dt.tz_localize(None)  # Remove timezone information
    df.columns = ['ds', 'y']

    # Train model
    model = Prophet()
    model.fit(df)
    st.write("Model trained successfully")

    # Predict the price for the next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

    # Plot the forecast
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

# Create the Streamlit GUI
st.title("Stock Price Predictor")

tickerSymbol = st.text_input("Enter Ticker Symbol:")
if st.button("Predict Price"):
    predict_price(tickerSymbol)