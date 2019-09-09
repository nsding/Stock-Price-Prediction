import math
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pandas import Series, DataFrame
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
from datetime import timedelta




start = datetime.datetime(2017, 1, 1)
end = datetime.datetime(2019, 8, 31)

df = web.DataReader("AAPL", 'yahoo', start, end)
print (df.tail())

df = df[['Open',  'High',  'Low',  'Adj Close', 'Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
forecast_col = 'Adj Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.1 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label','Adj Close'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

#Linear Regression
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

#Lasso
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train,y_train)  

confidence = clf.score(X_test, y_test)
print ("Linear Regression Confidence Level:",confidence)
confidencepoly2 = clfpoly2.score(X_test,y_test)
print ("Quadratic Regression 2 Confidence Level:",confidencepoly2)
confidencepoly3 = clfpoly3.score(X_test,y_test)
print ("Quadratic Regression 3 Confidence Level:",confidencepoly3)
confidencelasso= reg.score(X_test,y_test)
print ("Lasso Confidence Level:",confidencelasso)



#Show the Stock Price Prediction using Linear Regression
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

style.use('ggplot')
df['Adj Close'].plot()
df['Forecast'].plot()
#print (df['Forecast'].count())
plt.title("Stock Price Prediction using Linear Regression Method")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#Show the Stock Price Prediction using Quadractic Regression
forecast_quadractic = clfpoly2.predict(X_lately)
df['Forecast_Q'] = np.nan

last_date_1 = df.iloc[-df['Forecast'].count()].name
last_unix_1 = last_date_1
next_unix_1 = last_unix_1 + timedelta(days=1)

for i in forecast_quadractic:
    next_date_1 = next_unix_1
    next_unix_1 += timedelta(days=1)
    df.loc[next_date_1] = [np.nan for _ in range(len(df.columns)-1)]+[i]

style.use('ggplot')
df['Adj Close'].plot()
df['Forecast_Q'].plot()
plt.title("Stock Price Prediction using Quadractic Regression Method")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#Show the Stock Price Prediction using Lasso Regression
forecast_lasso = reg.predict(X_lately)
df['Forecast_lasso'] = np.nan

last_date_2 = df.iloc[-df['Forecast_Q'].count()].name
last_unix_2 = last_date_2
next_unix_2 = last_unix_2 + timedelta(days=1)

for i in forecast_lasso:
    next_date_2 = next_unix_2
    next_unix_2 += timedelta(days=1)
    df.loc[next_date_2] = [np.nan for _ in range(len(df.columns)-1)]+[i]

style.use('ggplot')
df['Adj Close'].plot()
df['Forecast_lasso'].plot()
plt.title("Stock Price Prediction using Lasso Regression Method")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
