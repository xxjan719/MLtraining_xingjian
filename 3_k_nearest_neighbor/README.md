## Implementing K-nearest neighbours for flower classification
@Jianxing add notes about KNN
### K-nearest neighbours
K is a hyperparameter need tuning. (Maybe we can take Sqrt(n), n is the number of data points. 
Usually we choose K as an odd value to avoid two classes have the same "votes")

Be aware that in Knn, there would not be parameters need updating. The distance function, K value are 
all decided.



### Assignment
``` Python
import numpy as np
from sklearn import datasets

# set up the data for the KNN model
X, y =datasets.load_iris(return_X_y=True)

test_size = 25

## Randomly shuffle data
np.random.seed(1)
mask = np.random.permutation(range(len(X)))
X = X[mask]
y = y[mask].reshape(-1,1)

## Assign to training and testing sets
X_train = X[test_size:,:]
y_train = y[test_size:,:]
X_test  = X[:test_size,:]
y_test  = y[:test_size,:]

# Build the KNN model
def knn_predict(X_train, y_train, X_test, k=5):

    return predictions

def accuracy(prediction, actual):
    return np.sum(prediction==actual)/len(prediction)

prediction = knn_predict(X_train, y_train, X_test, k=10)
print('Accuracy on test set: {:.2f}%'.format(accuracy(prediction, y_test)))
```
