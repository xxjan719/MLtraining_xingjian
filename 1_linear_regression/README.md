## Linear regression
@Ruitai, try to present some detials about linear regression here.

### Assignment
This task is to implement the linear regression includes two parts.
The first file is `linear_regression.py`, which is the algorithm object.
```Python
import numpy as np

class LinearRegression:
    def __init__(self, M, N):
        ''' param:
            M:    the number of data
            N:    the number of features
        '''
        self.M =
        self.N = 
        self.coef_ = 
        self.intercept_ = 

    def fit(self, X, y):
        '''
            update the coefficient of the linear regression model
            X: (#data, #feature)
            y: (#data, )
        '''

        return

    def predict(self, X):

        return y
```
The second file is the testing file, named `test.py`.
Here, I only emphasize the dataset. 
The test set includes the last 50 samples, and the training set includes the rests.
After you finish your linear regression, 
present the mean squared error of predicted and true results.
Also, present the results by `sklearn.linear_model.LinearRegression`.
```
from sklearn import datasets
House_price_X, House_price_y = datasets.load_boston(return_X_y=True)
```

**Normalization**: it is very import to do the normalization. Try function `sklearn.preporcessing`
