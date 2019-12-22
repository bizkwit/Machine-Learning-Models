import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import pickle

# upload the data set and clean for the purpose
data_frame = pd.read_csv('breast-cancer-wisconsin.data')
data_frame.replace('?', -99999, inplace=True)
data_frame.drop(['id'], 1, inplace=True)

X = np.array(data_frame.drop(['class'], 1))
Y = np.array(data_frame['class'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

classifier = neighbors.KNeighborsClassifier(n_jobs=-1)
classifier.fit(X_train, Y_train)

with open('knearest.pickle', 'wb') as file:
    pickle.dump(classifier, file)

pickle_in = open('knearest.pickle','rb')
classifier = pickle.load(pickle_in)

accuracy = classifier.score(X_train, Y_train)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1], [8,4,1,3,1,2,3,3,4]])
prediction = classifier.predict(example_measures)
print(prediction)