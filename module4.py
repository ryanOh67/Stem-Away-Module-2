import pandas as pd
import torch
df = pd.read_csv('helloRyan.csv')

from simpletransformers.classification import ClassificationModel

# define hyperparameter
train_args ={
             "reprocess_input_data": True,

             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": 4}

# Create a ClassificationModel
from sklearn.model_selection import train_test_split
new_df = pd.DataFrame()
new_df['LengthofContent'] = df['LengthofContent']
train_df, test_df = train_test_split(new_df, test_size=0.10)

print('train shape: ',train_df.shape)
print('test shape: ',test_df.shape)
model = ClassificationModel(
  
    "bert", "bert-base-cased",
    num_labels=3,
    use_cuda = False,
    args=train_args
)
model.train_model(train_df)
class_list = ['Short', 'Medium', 'Long']

post = "Short"

predictions, raw_outputs = model.predict([post])
from sklearn.metrics import f1_score, accuracy_score


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')
    
result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)
print(result)
print(class_list[predictions[0]])


 