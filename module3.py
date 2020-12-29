import logging
import pandas as pd


data = pd.read_csv('helloRyan.csv')
data.head()
print('hi')
print(data['Date'][2])
df = data.copy()
test_df = df.copy()
print_plot(10)

# go to the bug thing, stage change, and look at the blue bar on the botrtom of the  screen to push changes