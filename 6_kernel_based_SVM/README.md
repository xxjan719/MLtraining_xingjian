## kernel-based SVM


``` Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def printImage(image):
    fig = plt.figure
    plt.imshow(image, cmap='gray')
    plt.show()

def accuracy(ypred, yexact):
    p = np.array(ypred==yexact, dtype=int)
    return np.sum(p)/float(len(yexact))

X_train = pd.read_csv('Digits_X_train.csv').values
y_train = pd.read_csv('Digits_y_train.csv').values
X_test  = pd.read_csv('Digits_X_test.csv').values
y_test  = pd.read_csv('Digits_y_test.csv').values

# one-hot encoding
lb = preprocessing.LabelBinarizer(neg_label=-1)
lb.fit(y_train)
y_train_ohe = lb.transform(y_train)
y_test_ohe = lb.transform(y_test)

class SVM():
    def __init__(self, X, y, kernel='linear', lr=0.01, lmd=0.01, epoch=2000):
        '''
        - X: shape (M, N)
        - y: shape (M, P), P=10
        - W: shape (M, P)
        '''
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(X)
    
        self.X     = self.scaler.transform(X)
        self.y     = y
        self.M     = X.shape[0]
        self.N     = X.shape[1]
        self.P     = y.shape[1]
        self.W     = np.random.randn(self.M, self.P)
        self.b     = np.zeros((1, self.P))
        self.lr    = lr
        self.lmd   = lmd
        self.epoch = epoch

        self.kernels = {'linear': self.linear,
                        'rbf': self.rbf}
        self.kernel = self.kernels[kernel]
        self.gamma = 1/self.N
        self.coef0 = 0.
        self.KerMat = self.kernel_matrix(self.X)

    def kernel_matrix(self, X):
        # code the kernel method: try rbf kernel
        # check the following linear kernel
        return KerMat

    def linear(self, Xi, Xj):
        return np.dot(Xi, Xj)

    def rbf(self, Xi, Xj):
        return np.exp(-self.gamma*np.dot(Xi-Xj, Xi-Xj))

    def fit(self):
        for _ in range(self.epoch):
            # calculate the predicted values

            # calculate the condition

            # calculate the derivative

            self.W -= self.lr*dW
            self.b -= self.lr*db

    def predict(self, X_test):
        X = self.scaler.transform(X_test)
        KerMat = self.kernel_matrix(X)
        y_hat_test = np.dot(KerMat, self.W) + self.b
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ypred = np.zeros(X_test.shape[0], dtype=int)
        for i in range(X_test.shape[0]):
            ypred[i] = labels[np.argmax(y_hat_test[i, :])]
        return ypred

#mySVM = SVM(X_train, y_train_ohe, 'rbf')
#mySVM.fit()
#y_pred = mySVM.predict(X_test)
#print('Accuracy: ', accuracy(y_pred, y_test.ravel()))

from sklearn import svm
clf = svm.SVC(gamma='auto')
clf.fit(X_train, y_train.ravel())
y_hat = clf.predict(X_test)
print('before normalization, Acc. sklearn: ', accuracy(y_hat, y_test.ravel()))

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
clf1 = svm.SVC(gamma='auto')
clf.fit(X_train, y_train.ravel())
y_hat = clf.predict(X_test)
print('after normalization, Acc. sklearn: ', accuracy(y_hat, y_test.ravel()))
```
