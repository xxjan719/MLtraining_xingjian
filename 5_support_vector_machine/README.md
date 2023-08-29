## Support Vector Machine


``` Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys

def printImage(image):
    fig = plt.figure
    plt.imshow(image, cmap='gray')
    plt.show()

def accuracy(ypred, yexact):
    p = np.array(ypred==yexact, dtype=int)
    return np.sum(p)/float(len(yexact))

X_train = pd.read_csv('../Data/Digits_X_train.csv').values
y_train = pd.read_csv('../Data/Digits_y_train.csv').values
X_test  = pd.read_csv('../Data/Digits_X_test.csv').values
y_test  = pd.read_csv('../Data/Digits_y_test.csv').values
printImage(X_train[0].reshape(8,8))
sys.exit()

#scaler = preprocessing.StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test  = scaler.transform(X_test)

# one-hot encoding
lb = preprocessing.LabelBinarizer(neg_label=-1)
lb.fit(y_train)
y_train_ohe = lb.transform(y_train)
y_test_ohe = lb.transform(y_test)

class SVM():
    def __init__(self, X, y, lr=0.01, lmd=0.01, epoch=2000):
#    '''
#    - X: shape (M, N)
#    - y: shape (M, P), P=10
#    - W: shape (N, P)
#    '''
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(X)
    
        self.X     = self.scaler.transform(X)
        self.y     = y
        self.M     = X.shape[0]
        self.N     = X.shape[1]
        self.P     = y.shape[1]
        self.W     = np.random.randn(self.N, self.P)
        self.b     = np.zeros((1, self.P))
        self.lr    = lr
        self.lmd   = lmd
        self.epoch = epoch

    def fit(self):
        for _ in range(self.epoch):
            # calculate the predicted values

            # calculate the condition

            # calculate the derivative

            self.W -= self.lr*dW
            self.b -= self.lr*db

    def predict(self, X_test):
        X = self.scaler.transform(X_test)
        y_hat_test = np.dot(X, self.W) + self.b
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ypred = np.zeros(X_test.shape[0], dtype=int)
        for i in range(X_test.shape[0]):
            ypred[i] = labels[np.argmax(y_hat_test[i, :])]
        return ypred

mySVM = SVM(X_train, y_train_ohe)
mySVM.fit()
y_pred = mySVM.predict(X_test)
print('Accuracy: ', accuracy(y_pred, y_test.ravel()))

from sklearn import svm
clf = svm.LinearSVC()
clf.fit(X_train, y_train.ravel())
y_hat = clf.predict(X_test)
print('Acc. sklearn: ', accuracy(y_hat, y_test.ravel()))
```
