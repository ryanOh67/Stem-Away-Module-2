import pandas as pd

df = pd.read_csv('helloRyan.csv')

from simpletransformers.classification import ClassificationModel

# define hyperparameter
train_args ={"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": 4}

# Create a ClassificationModel

model = ClassificationModel(
    "bert", "bert-base-cased",
    num_labels=12,
    args=train_args
)
model.train_model(train_df)

