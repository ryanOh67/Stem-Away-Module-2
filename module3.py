
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
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

#%matplotlib inline


data = pd.read_csv('helloRyan.csv', dtype = str)
data.head()

df = data.copy()
X = df.Likes
y = df.LengthofContent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42) #when I plug in 0.5 for the test size, it works with values

X_train.head()  #line 113
y_train.head()

#print(y_train)
#categorize to long, medium short?
nb = Pipeline([('vect', CountVectorizer(lowercase=False)),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)



y_pred = nb.predict(X_test)

#relationship between the length of the post and the likes it gets

my_categories = ['Short', 'Medium', 'Long']

#'38','208','356','60', '1504','861','27','215'
print('accuracy %s' % accuracy_score(y_pred, y_test))
res1311 = accuracy_score(y_pred, y_test)
#print(len(y_pred))
#print(len(y_test))
#print(len(y_train))
print(classification_report(y_test, y_pred,labels =['Short', 'Medium', 'Long'], target_names=my_categories)) #define my_categories
#labels=[1, 2, 3,4,5,6,7,8,9,10,11]



# go to the bug thing, stage change, and look at the blue bar on the botrtom of the  screen to push changes
#recommends the length of the post?z
