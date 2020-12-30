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
df = data.copy()

print(df.info())


my_categories = ['post']
plt.figure(figsize=(10,4))
df.Category.value_counts().plot(kind='bar')


# go to the bug thing, stage change, and look at the blue bar on the botrtom of the  screen to push changes