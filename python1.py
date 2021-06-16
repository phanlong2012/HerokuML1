import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('FileCSV/houseprice1.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

def convertInt(word):
    wordDict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
                'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10,
                'eleven':11, 'twelve':12, 'zero':0, 0:0}
    return wordDict[word]

X = dataset.iloc[:, :3]

X['experience'] = X['experience'].apply(lambda x:convertInt(x))

y = dataset.iloc[:, -1]

print(X)
print(y)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor, open('SavedModel/model.pkl','wb'))

model = pickle.load(open('SavedModel/model.pkl','rb'))

print(model.predict([[2,9,6]]))



