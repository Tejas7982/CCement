import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot 
from sqlalchemy import create_engine
import numpy as np

df = pd.read_csv(r"C:/Users/Tejas/Downloads/Final_Cement_Dataset (1).csv")
df.info()
df.columns
df.drop(columns={'Cement Production','month','limestone','Demand','GDP_Construction_Rs_Crs','Trasportation_Cost'},inplace =True)

from statsmodels.tsa.stattools import adfuller
def ad_test(df):
    dftest = adfuller(df, autolag = 'AIC')
    print("1. ADF                :    ",dftest[0])
    print("2. P-Value            :    ",dftest[1])
    print("3. No. of Lags        :    ",dftest[2])
    print("4. No of Observations :    ",dftest[3])
    print("5. Critical Values    :    ")
    for key, val in dftest[4].items():
            print("\t", key,":", val)
            
ad_test(df['Sales'])

# Data Partition
Train = df.head(131)
Test = df.tail(12)


df1 = pd.read_csv('C:/Users/Tejas/OneDrive/Desktop/Final Deployment/test_arima.csv') 

###############prediction for Sales##################

tsa_plots.plot_acf(df.Sales, lags = 12)
tsa_plots.plot_pacf(df.Sales, lags = 12)


# ARIMA with AR = 4, MA = 6
model1 = ARIMA(Train.Sales, order = (12, 1, 6))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
start_index
end_index = start_index + 11
forecast_test = res1.predict(start = start_index, end = end_index)

print(forecast_test)

model1 = model1.fit()

train_forecast = Train.copy()
test_forecast = Test.copy()
train_forecast['forecasted'] = model1.predict()
train_forecast[['Sales','forecasted']].plot(figsize=(20,8))



# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.Sales, forecast_test))
print('rmse_ARIMA: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.Sales)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()

from sklearn.metrics import mean_absolute_error
#MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true))
mape_ARIMA = mean_absolute_percentage_error(Test.Sales, forecast_test)
print('mape_ARIMA : ',mape_ARIMA)

#MAE
mae_ARIMA = mean_absolute_error(Test.Sales, forecast_test)
print('mae_ARIMA : ',mae_ARIMA)

# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm
help(pm.auto_arima)

ar_model = pm.auto_arima(Train.Sales, start_p = 0, start_q = 0,
                      max_p = 16, max_q = 16, # maximum p and q
                      m = 1,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = False,   # No Seasonality
                      start_P = 0, trace = True,
                      error_action = 'warn', stepwise = True)


# Best Parameters ARIMA
# ARIMA with AR=3, I = 1, MA = 5
model = ARIMA(Train.Sales, order = (4,1,1))
res = model.fit()
print(res.summary())


# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res.predict(start = start_index, end = end_index)


print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.Sales, forecast_best))
print('rmse_Auto-ARIMA: %.3f' % rmse_best)
# plot forecasts against actual outcomes
pyplot.plot(Test.Sales)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true))
mean_absolute_percentage_error(Test.Sales, forecast_best)
print('mape_Auto-ARIMA : ',mape_ARIMA)

#MAE
mean_absolute_error(Test.Sales, forecast_best)
print('mae_Auto-ARIMA : ',mae_ARIMA)

out = model.plot_diagnostics(figsize=(10,8))


# saving model whose rmse is low
# The models and results instances all have a save and load method, so you don't need to use the pickle module directly.
# to save model
res1.save("model3.pickle")
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model3.pickle")

# Forecast for future 12 months
start_index = len(df)
start_index
end_index = start_index + 11
forecast = model.predict(start = start_index, end = end_index)

print(forecast)
pyplot.plot(Test.Sales)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()

import os
cwd = os.getcwd()
