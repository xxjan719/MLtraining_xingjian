## Logistic regression
@Xingjian, try to present detials about logistic regression

### Assignment
This taks is to implement the logistic regression similiar as what we did to linear regression.
The first file is `logistic_regression.py`, which is the algorithm main body.
```Python
import numpy as np
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    # write the sigmoid function
    return

class LogisticRegression:
    def __init__(self, M, N, lr=0.1):
        # unlike linear regression. we need the weight matrix here
        # StandardScaler(): normalizer, which is important
        # initailize all variables
        self.W = # weight matrix
        self.b = # bias

    def fit(self, X, y, epoch=2000):
        # normalize first
        # for each epoch, update the weight matrix and bias

    def predict(self, X_test):

        return y_pred
```
This time, the test function is included in `.`.
