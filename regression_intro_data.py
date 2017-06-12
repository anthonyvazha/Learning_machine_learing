import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
#make plots look good
style.use('ggplot')
#getting the stock information(google) form quandl
#quandl does have a limt of how many # access their tokens without making account
df = quandl.get('WIKI/GOOGL')
print(df.tail())



# the times * 100 is for us, it doesn't really matter for the classifier
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#Voltility percentage - Formula = (High-Low)/(Low)*100
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100.0
#Daily percentage change - Formula = (New - old)/(old)*100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
#all the things we actully care about
#Volume defintion is the basicaly how many trades happend in that day
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
#fill the blank spots or missing data with -99999, na is panda term for missing data
#Machine learning you cant have missing data
#99999 is treated like outlier instead of getting rid of rows/columns
df.fillna(-99999, inplace=True)

#math.ceil just rounds everthing up if a decimal value also a float
#the .1 is to predict 10% of data frame(dates, so 100 days = days into the future
forecast_out = int(math.ceil(0.01*len(df)))
#make a label columb---> this shifts using panda function
#based on forecast_out number of days head for better train the algorithm
df['label']= df[forecast_col].shift(-forecast_out)

#features are labeled as X = drop the labels column only,
#labels are labeled as Y = is only the label column
#is being converted to a numpy array
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#scaling the features between 1 and -1
X = preprocessing.scale(X)
#drop all the na terms
df.dropna(inplace=True)
y = np.array(df['label'])
#print(len(x),len(y))
#shuffle your data(with same row) for not having bias of the sample
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)
#train and test on diffrent copies of the data so there is no bias
#training you data
#n_jobs is for running multiple threads as possible - makes training faster
clf = LinearRegression(n_jobs=-1)
#switch to support vector machine
#clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
#saving the classifier
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)
#use the saved classifier - this for in case the classifier to big to run all the time
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
#test you data
#this prints out a really high accuracy which is proboably due to the fact of the data
#label data ranges form low number like in the 40s to high 800s and has high variance
#even for suffled data set
accuracy= clf.score(X_test,y_test)
#print(accuracy)
#this is predicting the future stock closing prices
forecast_set = clf.predict(X_lately)

#print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
#dates
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
#plot
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('Price')
plt.show()




#print both to see data
#print(df.head())
#print(df.tail())
