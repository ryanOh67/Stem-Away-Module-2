
import re
from bs4 import BeautifulSoup
import logging
import pandas as pd
import numpy as np
from numpy import random
#import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
#%matplotlib inline


data = pd.read_csv('helloRyan.csv')
data.head()
#print('hi')
#print(data['Date'][2])
df = data.copy()

#print(df.info())

X = df.Length
y = df.Likes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

X_train.head()  #line 113
y_train.head()

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)


from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)


#relationship between the length of the post and the likes it gets

my_categories = ['44', '47', '98', '38', '208','356', '60', '1504', '861','27','215']


print('accuracy %s' % accuracy_score(y_pred, y_test))
res1311 = accuracy_score(y_pred, y_test)
print(classification_report(y_test, y_pred,target_names=my_categories)) #define my_categories

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

# go to the bug thing, stage change, and look at the blue bar on the botrtom of the  screen to push changes
#recommends the length of the post?
