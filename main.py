import wandb
import pandas as pd
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchmetrics

# Encode target labels with value between 0 and n_classes-1.
# Import Metrics for use with evaluation

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Split dataset in train and test with a ratio of 70-30

from sklearn.model_selection import train_test_split

# set device to cuda if available else pass to cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %env "WANDB_NOTEBOOK_NAME" "demo_wine_wandb_test"
wandb.login()

# SET SEED
torch.manual_seed(32)
np.random.seed(32)
torch.use_deterministic_algorithms(True)

df = pd.read_csv("./data/wine_data.csv")
df.head()

le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])

# set the feature variables

df_features = df.drop('Class', axis=1)
df_features.head()

# Set the target variable

df_target = df[['Class']]
df_target.head()

# Split the dataset
X_train, x_test, Y_train, y_test = train_test_split(df_features,
                                                    df_target,
                                                    test_size=0.3,
                                                    random_state=42)


X_train.shape, x_test.shape,  Y_train.shape, y_test.shape,

Xtrain = torch.from_numpy(X_train.values).float()
Xtest = torch.from_numpy(x_test.values).float()
print(Xtrain.shape, Xtest.shape)

print(Xtrain.dtype, Xtest.dtype)

# Reshape tensor to 1D

Ytrain = torch.from_numpy(Y_train.values).view(1, -1)[0]
Ytest = torch.from_numpy(y_test.values).view(1, -1)[0]
print(Ytrain.shape, Ytest.shape)

input_size = 13
output_size = 3
hidden_size = 100

config = dict(
    input_size=13,
    output_size=3,
    hidden_size=100,
    dataset="wine dataset",
    architecture='Linear',
    onnx_model_path="/models/wine_model.onnx",
    learning_rate=0.01,
    # CHANGE THE LOSS
    # loss=nn.NLLLoss(),
    loss=nn.CrossEntropyLoss(),
    # CHANGE THE OPTIMIZER
    # optimizer="adam",
    optimizer="SGD",
    # optimizer="adagrad"
)
for k, v in config.items():
    print(f"wandb config{k}:{v}")

# Define Class Net


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # ADD DROPOUT
        # self.dropout = nn.Dropout(p=0.25)  # DROPOUT

    def forward(self, X):
        X = torch.sigmoid((self.fc1(X)))
        # X = self.dropout(X)  # DROPOUT
        X = torch.sigmoid(self.fc2(X))
        # X = self.dropout(X)  # DROPOUT
        X = self.fc3(X)

        return F.log_softmax(X, dim=-1)


# Instantiate the model
model = Net()
# preview out model
print(model)

optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate"))
loss_fn = config.get("loss")

# TRAINING LOOP
# TRAIN THE MODEL

epochs = 1000
with wandb.init(project="demo_wandb_sklearn", config=config):
    wandb.watch(model, criterion=None, log="gradients", log_freq=10)

    for epoch in range(epochs):

        optimizer.zero_grad()
        Ypred = model(Xtrain)

        loss = loss_fn(Ypred, Ytrain)
        acc = torchmetrics.functional.accuracy(Ypred, Ytrain)
        loss.backward()

        optimizer.step()

        wandb.log(
            {"Train": {'Epoch': epoch, "Loss": loss.item(), "Accuracy": acc}})

    # SAVE MODEL STATE DICT TO DISK

    wandb.save(torch.save(model.state_dict(), "./models/wine_train.pt"))

    # LOAD MODEL FROM DISK and EVALUATE

    new_model = Net()
    new_model.load_state_dict(torch.load("./models/wine_train.pt"))
    new_model.eval()

    # SET THE PREDICTIONS

    predict = new_model(Xtest)
    _, predict_y = torch.max(predict, 1)

    # VISUALIZE CONFUSION MATRIX

    wandb.sklearn.plot_confusion_matrix(Ytest, predict_y, labels=[0, 1, 2])

    # Print Metrics

    wandb.log({"Test": {"accuracy_score": accuracy_score(Ytest, predict_y),
               "precision_score": precision_score(Ytest, predict_y, average='weighted'),
                        "recall_score": recall_score(Ytest, predict_y, average="weighted")}})

    table = wandb.Table(data=df, columns=[df_features, df_target])
    wandb.log({"Data Table": table})

    torch.onnx.export(model=model, args=(Xtrain), f="./models/wine_test.onnx", input_names=['input'], output_names=['output'],
                      verbose=True, do_constant_folding=True, opset_version=11)
    # COPY ONNX TO WANDB RUN DIR FOR LOGGING
    shutil.copy("./models/wine_test.onnx",
                os.path.join(wandb.run.dir,
                             "wine_test.onnx"))
    # COPY PT TO WANDB RUN DIR FOR LOGGING
    shutil.copy("./models/wine_train.pt",
                os.path.join(wandb.run.dir, "wine_train.pt"))

wandb.finish()
torch.cuda.empty_cache()

# END OF FILE
