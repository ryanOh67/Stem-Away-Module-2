import logging
import pandas as pd


data = pd.read_csv('helloRyan.csv')
data.head()
print('hi')
print(data['Date'][2])