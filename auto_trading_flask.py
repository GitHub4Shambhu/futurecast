import pandas as pd
import yfinance as yf
from prophet import Prophet
from flask import Flask, request, render_template_string
import matplotlib.pyplot as plt
import io
import base64
from lstm_model import get_stock_data, prepare_data, build_model, predict_price
import datetime
import seaborn as sns
import numpy as np

# Set the Matplotlib backend to Agg
plt.switch_backend('Agg')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_stock_price():
    if request.method == 'POST':
        tickerSymbol = request.form['ticker']
        if not tickerSymbol:
            return render_template_string(template, error="Please enter a ticker symbol")

        # Get data on this ticker
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=5*365)
        tickerData = yf.Ticker(tickerSymbol)
        tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

        if (tickerDf.empty):
            return render_template_string(template, error="No data found for the ticker symbol")

        # Prepare data for Prophet
        df = tickerDf.reset_index()[['Date', 'Close']]
        df['Date'] = df['Date'].dt.tz_localize(None)  # Remove timezone information
        df.columns = ['ds', 'y']

        # Train model
        model = Prophet()
        model.fit(df)

        # Predict the price for the next 30 business days in the future
        future_dates = pd.bdate_range(start=end_date, periods=30).to_frame(index=False, name='ds')
        forecast = model.predict(future_dates)
        forecast_data = forecast[['ds', 'yhat']]
        forecast_data.columns = ['Prediction Date', 'Predicted Price']
        forecast_data['Predicted Price'] = forecast_data['Predicted Price'].round(2)
        forecast_data = forecast_data.to_html(classes='table table-striped', index=False)

        # Plot the forecast with Seaborn for better aesthetics
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot actual prices for the last 30 days
        actual_prices = df.tail(30)
        sns.lineplot(x=actual_prices['ds'], y=actual_prices['y'], label='Actual Price', color='blue', ax=ax)
        # Plot predicted prices for the next 30 business days
        forecast_plot = forecast[['ds', 'yhat']].tail(30)
        sns.lineplot(x=forecast_plot['ds'], y=forecast_plot['yhat'], label='Predicted Price', color='red', ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Stock Price Prediction')
        ax.legend()
        plt.xticks(rotation=45)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # LSTM model prediction
        lstm_data = get_stock_data(tickerSymbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        x_train, y_train, scaler, training_data_len, dataset = prepare_data(lstm_data)
        lstm_model = build_model()
        lstm_model.fit(x_train, y_train, batch_size=1, epochs=1)
        lstm_predictions, y_test = predict_price(lstm_model, scaler, dataset, training_data_len)
        print("LSTM Predictions:", lstm_predictions)
        print("Actual Prices:", y_test)
        # Plot actual vs. predicted prices for LSTM model
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(len(y_test)), y=y_test.flatten(), label='Actual Price', color='blue', ax=ax)
        sns.lineplot(x=range(len(lstm_predictions)), y=lstm_predictions.flatten(), label='Predicted Price', color='red', ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title('LSTM Model: Actual vs. Predicted Prices')
        ax.legend()
        plt.xticks(rotation=45)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        lstm_plot_url = base64.b64encode(img.getvalue()).decode()
        lstm_forecast_dates = pd.bdate_range(start=end_date, periods=30).to_frame(index=False, name='Date')
        lstm_predictions = lstm_predictions[-30:].flatten()
        lstm_predictions = np.round(lstm_predictions, 2)
        lstm_forecast_data = pd.DataFrame({'Date': lstm_forecast_dates['Date'], 'Predicted Price': lstm_predictions})
        lstm_forecast_data['Predicted Price'] = lstm_forecast_data['Predicted Price'].apply(lambda x: f'{x:.2f}')
        lstm_forecast_data = lstm_forecast_data.to_html(classes='table table-striped', index=False)

        return render_template_string(template, ticker=tickerSymbol, forecast_data=forecast_data, plot_url=plot_url, lstm_forecast_data=lstm_forecast_data, lstm_plot_url=lstm_plot_url)

    return render_template_string(template)

# Updated template with Bootstrap
template = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <title>Stock Price Predictor</title>
    <style>
        body { padding-top: 56px; }
        .container { max-width: 960px; }
        .card { margin-top: 20px; }
        .table-responsive { margin-top: 20px; }
        .img-fluid { margin-top: 20px; }
        .card-header h1 { text-align: center; }
        table { font-family: Arial, sans-serif; border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #dddddd; text-align: left; padding: 8px; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .progress { margin-top: 20px; }
    </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark bg-primary fixed-top">
    <div class="container">
        <a class="navbar-brand mx-auto" href="#">Future Forecaster</a>
    </div>
</nav>
<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h1>Stock Price Predictor</h1>
                </div>
                <div class="card-body">
                    <form method="post">
                        <div class="form-group">
                            <label for="ticker">Enter Ticker Symbol (To Predict Future Price):</label>
                            <input type="text" class="form-control" name="ticker" id="ticker" required>
                        </div>
                        <button type="submit" class="btn btn-primary btn-block">Predict</button>
                    </form>
                    <div class="progress">
                        <div class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                    {% if error %}
                    <div class="alert alert-danger mt-3">{{ error }}</div>
                    {% endif %}
                    {% if forecast_data %}
                    <h2 class="mt-4">Forecast for {{ ticker }} for the next 30 days</h2>
                    <div class="table-responsive">
                        {{ forecast_data | safe }}
                    </div>
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Forecast plot" class="img-fluid">
                    <h2 class="mt-4">LSTM Forecast for {{ ticker }} for the next 30 days</h2>
                    <div class="table-responsive">
                        {{ lstm_forecast_data | safe }}
                    </div>
                    <img src="data:image/png;base64,{{ lstm_plot_url }}" alt="LSTM Forecast plot" class="img-fluid">
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</div>
<script>
    document.querySelector('form').addEventListener('submit', function(event) {
        event.preventDefault();
        var form = this;
        var button = form.querySelector('button[type="submit"]');
        var progressBar = document.querySelector('.progress-bar');
        button.disabled = true;
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        var xhr = new XMLHttpRequest();
        xhr.open('POST', form.action, true);
        xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4 && xhr.status === 200) {
                document.body.innerHTML = xhr.responseText;
                button.disabled = false;
            }
        };
        xhr.upload.onprogress = function(event) {
            if (event.lengthComputable) {
                var percentComplete = (event.loaded / event.total) * 100;
                progressBar.style.width = percentComplete + '%';
                progressBar.setAttribute('aria-valuenow', percentComplete);
            }
        };
        var formData = new FormData(form);
        xhr.send(new URLSearchParams(formData).toString());
    });
</script>
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
