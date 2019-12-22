import math, datetime, pickle
import pandas as pd
import quandl  # Financial datasets. can go on the wegsite to find specific data and get the python code.
import numpy as np
from sklearn import preprocessing  # will use to scale all the feature data to be between 1 and -1
from sklearn import model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style


style.use('ggplot')

quandl.ApiConfig.api_key = "fEioLEE1s6CN3ymzZQri"
df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# to calculate the margin between high and low
df['HL_percent'] = (df["Adj. High"] - df['Adj. Low']) / df['Adj. Low'] * 100.0

# to calculate the change between the opening price and the closing price
df['percent_change'] = (df["Adj. Close"] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_percent', 'percent_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)  # fill with this in case data is missing since cannot work with NaN

forecast_out = int(math.ceil(0.1 * len(df)))  # 0.1 to predict today's price form data that came 10 days ago

df['label'] = df[forecast_col].shift(-forecast_out)

# for features, create an array without the label
X = np.array(df.drop(['label'], 1))
# makes it easier to training and testing but adds much more processing time. don't use if time is of the essence
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]  # substring from the forecast to the end
X = X[:-forecast_out]  # substring up to the forecast


df.dropna(inplace=True)
# for labels
Y = np.array(df['label'])

# creating test and train sets out of the data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

classifier = LinearRegression(n_jobs=-1)  # number of threads for training (-1 to run on max CPU usage)
classifier.fit(X_train, Y_train)

# to serialize the classifier
with open('linearregression.pickle', 'wb') as file:
    pickle.dump(classifier, file)
# to load the serialized classifier
pickle_in = open('linearregression.pickle', 'rb')
classifier = pickle.load(pickle_in)

accuracy = classifier.score(X_test, Y_test)
# print("Days: ", forecast_out, " Accuracy: ", accuracy)

forecast_set = classifier.predict(X_lately)

# print(forecast_set, accuracy, forecast_out)

# create vector with NaNs
df['forecast'] = np.nan
# creating dares since its not part of our data and not a feature for prediction
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 # number of seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df["Adj. Close"].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


