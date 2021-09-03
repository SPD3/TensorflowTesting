from math import isnan
import tensorflow as tf
import numpy as np 
import pandas as pd
import os

train_data = pd.read_csv("/Users/seandoyle/git/TensorflowTesting/titanic/train.csv")

def preProcessData(data, shouldSeperateLabelsFromData=True):
    data = data.to_numpy()
    data = eliminateFirstColumn(data)
    if(shouldSeperateLabelsFromData):
        y, X = seperateLabelsFromData(data)
    else:
        y = None
        X = data
    X = scaleX(X)
    
    return y, X

def eliminateFirstColumn(data):
    newData = []
    for line in data:
        newData.append(line[1:])
    return newData

def seperateLabelsFromData(data):
    y = []
    X = []
    for line in data:
        y.append(line[0])
        X.append(line[1:])
    return y, X

def scaleX(X):
    X = scalePClass(X)
    X = eliminateName(X)
    X = binarySex(X)
    X = scaleAge(X)
    X = scaleParch(X)
    X = eliminateTicketNumber(X)
    X = scaleFare(X)
    X = removeLastTwoColumns(X)
    replaceNans(X)
    return X

def prettyPrint(X):
    length = 4
    if length > len(X):
        length = len(X)
    for i in range(length):
        print(X[i])

def scalePClass(X):
    return scaleClass(X, 0, 1.0/3.0)

def scaleClass(X, index, amount):
    newX = []
    for passenger in X:
        passenger[index] *= amount
        newX.append(passenger)
    return newX

def eliminateName(X):
    newX = []
    for passenger in X:
        newPassenger = []
        newPassenger.append(passenger[0])
        otherNecessaryItems = passenger[2:]
        for otherNecessaryItem in otherNecessaryItems:
            newPassenger.append(otherNecessaryItem)
        newX.append(newPassenger)
    return newX

def binarySex(X):
    newX = []
    for passenger in X:
        if(passenger[1] == "male"):
            passenger[1] = 1.0
        else:
            passenger[1] = 0.0
        newX.append(passenger)
    return newX

def scaleAge(X):
    return scaleClass(X, 2, 1.0/100.0)

def scaleSibSp(X):
    return scaleClass(X, 3, 1.0/8.0)

def scaleParch(X):
    return scaleClass(X, 4, 1.0/6.0)

def eliminateTicketNumber(X):
    newX = []
    for passenger in X:
        newPassenger = []
        firstItems = passenger[:5]
        lastItems = passenger[6:]
        for item in firstItems:
            newPassenger.append(item)
        for item in lastItems:
            newPassenger.append(item)
        newX.append(newPassenger)
         
    return newX

def scaleFare(X):
    return scaleClass(X, 5, 1.0/512.0)

def removeLastTwoColumns(X):
    newX = []
    for passenger in X:
        newX.append(passenger[:-2])
    return newX

def replaceNans(X):
    for i in range(len(X)):
        for j in range(len(X[0])):
            if(np.isnan(X[i][j])):
                X[i][j] = 0

def createModel(shape):
    inputs = tf.keras.layers.Input(shape=(shape))
    layerSize = 1024
    x = tf.keras.layers.Dense(layerSize, activation="relu")(inputs)
    for i in range(5):
        x = tf.keras.layers.Dense(layerSize, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    return model

y, X = preProcessData(train_data)
model = createModel(len(X[0]))

X = np.array(X)
y = np.array(y)

checkpoint_path = "training_1/cp.ckpt2"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
if(True):
    model.fit(X, y, epochs=50, validation_split=0.1, callbacks=[cp_callback])
