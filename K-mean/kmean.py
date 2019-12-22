import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
style.use('ggplot')




# to serialize the classifier
with open('linearregression.pickle', 'wb') as file:
    pickle.dump(classifier, file)
# to load the serialized classifier
pickle_in = open('linearregression.pickle', 'rb')
classifier = pickle.load(pickle_in)