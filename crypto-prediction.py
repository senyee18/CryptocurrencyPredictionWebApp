import streamlit as st
from datetime import date

import pandas as pd
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Prediction App")

stocks = ("ETH-USD", "BTC-USD")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

# Preprocessing
df = data[['Date', 'Close']]
df = df.rename(columns={"Date": "ds", "Close": "y"})

# Train-test split
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Forecasting
m = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_mode='multiplicative',
    yearly_seasonality=10,
    weekly_seasonality=False,
    daily_seasonality=False,
    holidays_prior_scale=20
)
m.fit(train_data)

future = m.make_future_dataframe(periods=len(test_data), freq='D')
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Evaluation

# fig = m.plot(forecast)
# st.plotly_chart(fig)

# fig2 = m.plot_components(forecast)
# st.write(fig2)

# mape = (abs(test_data['y'] - forecast['yhat'][-len(test_data):]) / test_data['y']).mean()
# st.write(f'MAPE: {mape:.2%}') *#

# Evaluation
fig = m.plot(forecast)
plt.title('Forecast')
st.pyplot(fig)

fig2 = m.plot_components(forecast)
plt.suptitle('')
st.pyplot(fig2)

mape = (abs(test_data['y'] - forecast['yhat'][-len(test_data):]) / test_data['y']).mean()
st.write(f'MAPE: {mape:.2%}')