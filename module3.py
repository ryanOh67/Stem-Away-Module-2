
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
#%matplotlib inline


data = pd.read_csv('helloRyan.csv')
data.head()

df = data.copy()
X = df.Likes
y = df.LengthofContent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state = 50) #when I plug in 0.5 for the test size, it works with values

X_train.head()  
y_train.head()

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
my_categories = ['Short', 'Medium', 'Long']


#print('accuracy %s' % accuracy_score(y_pred, y_test))
res1311 = accuracy_score(y_pred, y_test)

#print(classification_report(y_test, y_pred,labels =['Short', 'Medium', 'Long'], target_names=my_categories)) #define my_categories


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state = 50)
X_train.head()  
y_train.head()

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=50, max_iter=6, tol=None)),
               ])

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)


#print('accuracy %s' % accuracy_score(y_pred, y_test))
res1321 = sgd.score(y_pred, y_test)

#print(classification_report(y_test, y_pred,labels =['Short', 'Medium', 'Long'],target_names=my_categories))







from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state = 50)
X_train.head()  
y_train.head()
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
res2331 = logreg.score(y_pred, y_test)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state = 50)
X_train.head()  
y_train.head()
dtree = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', DecisionTreeClassifier(random_state=0)),
               ])
dtree.fit(X_train, y_train)

y_pred = dtree.predict (X_test)
res1341 = accuracy_score(y_pred, y_test)
#print(classification_report(y_test, y_pred,labels =['Short', 'Medium', 'Long'],target_names=my_categories))

results = pd.DataFrame({'Model': ['Naive Bayes MultinomialNB', 'Linear SVM', 'Logistic Regression', 'Decision Tree'],
                         'Accuracy': [res1311,res1341, res1321, res2331]})
results.set_index('Model')
results.sort_values(by='Accuracy')
print(results)
 
# go to the bug thing, stage change, and look at the blue bar on the botrtom of the  screen to push changes
#recommends the length of the post?
