import logging
import pandas as pd
import numpy as np
from numpy import random
#import gensim
#import nltk
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
#from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
#%matplotlib inline


data = pd.read_csv('helloRyan.csv')
data.head()
print('hi')
print(data['Date'][2])
test_df = data.copy()

print(df.info())

X = test_df.post
y = test_df.Category
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

# go to the bug thing, stage change, and look at the blue bar on the botrtom of the  screen to push changes
#recommends the length of the post?
