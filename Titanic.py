import tensorflow as tf
import numpy as np 
import pandas as pd

print("Hello world")
train_data = pd.read_csv("/Users/seandoyle/git/TensorflowTesting/titanic/train.csv")
print(train_data.head())

test_data = pd.read_csv("/Users/seandoyle/git/TensorflowTesting/titanic/test.csv")

def preProcessData(data):
    data = data.to_numpy()
    data = eliminateFirstColumn(data)
    y, X = seperateLabelsFromData(data)
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
    print("AFTER EVERYTHING")
    prettyPrint(X)
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

y, X = preProcessData(train_data)

