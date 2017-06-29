import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from pandas.tools.plotting import scatter_matrix

churn = pd.read_csv("data/churn_train.csv")

churn['churn'] =1 * (churn.last_trip_date >= '2014-06-01')
churn.phone = pd.get_dummies(churn.phone) # Android: 1, IPhone: 0
churn.rename(index = str, columns = {'phone': 'Android'}, inplace=True)# change the name

churn = pd.concat([churn, pd.get_dummies(churn.city, drop_first=True)], axis=1)
churn.drop('city', inplace=True, axis = 1)

churn.dropna(how = 'any', inplace= True) # Drop NaN

# Get account age in days
#churn['acc_age'] = (pd.to_datetime(churn['last_trip_date']) - pd.to_datetime(churn['signup_date']) ) / np.timedelta64(1,'D')
#churn.drop(['signup_date','last_trip_date'], inplace=True, axis = 1)

# or rather:

pull_date = pd.to_datetime('2014-07-01')
churn['acc_age'] = (pull_date - pd.to_datetime(churn['signup_date']) ) / np.timedelta64(1,'D')
churn.drop(['signup_date','last_trip_date'], inplace=True, axis = 1)

# Dummy luxury car
churn['luxury_car_user'] = 1 * (churn['luxury_car_user'])

#scatter_matrix(churn, diagonal='kde', figsize=(15,10))
#plt.show()

y = churn.churn
#X = churn.drop(['churn', 'last_trip_date', 'signup_date'], axis = 1)
X = churn.drop('churn', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1)


# Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
mnb.score(X_test,y_test)

gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb.score(X_test,y_test)

'''Whole Dataset'''

df = pd.read_csv("data/churn_train.csv")

df['churn'] =1 * (df.last_trip_date >= '2014-06-01')
df.phone = pd.get_dummies(df.phone) # Android: 1, IPhone: 0
df.rename(index = str, columns = {'phone': 'Android'}, inplace=True)# change the name

df = pd.concat([df, pd.get_dummies(df.city, drop_first=True)], axis=1)
df.drop('city', inplace=True, axis = 1)

pull_date = pd.to_datetime('2014-07-01')
df['acc_age'] = (pull_date - pd.to_datetime(df['signup_date']) ) / np.timedelta64(1,'D')
df.drop(['signup_date','last_trip_date'], inplace=True, axis = 1)

# Dummy luxury car
df['luxury_car_user'] = 1 * (df['luxury_car_user'])

'''Rating Stuff:
'''
pd.DataFrame.hist(churn,'avg_rating_by_driver',normed=True,grid=False)
plt.show()
pd.DataFrame.hist(churn,'avg_rating_of_driver',normed=True,grid=False)
plt.show()

# Thresholds

rat_by_t = df['avg_rating_by_driver'].median(axis=0)
rat_of_t = df['avg_rating_of_driver'].median(axis=0)

df['Rate_by_High'] = 1 * (df['avg_rating_by_driver'] >= rat_by_t)
df['Rate_by_Low'] = 1 * (df['avg_rating_by_driver'] < rat_by_t)
df['Rate_of_High'] = 1 * (df['avg_rating_of_driver'] >= rat_of_t)
df['Rate_of_Low'] = 1 * (df['avg_rating_of_driver'] < rat_of_t)
df.drop(['avg_rating_by_driver','avg_rating_of_driver'], inplace=True, axis = 1)

y = df.churn
X = df.drop('churn', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1)
