import numpy as np
from sklearn import datasets

# set up the data for the KNN model
X, y = datasets.load_iris(return_X_y=True)

test_size = 25

## Randomly shuffle data
np.random.seed(1)
mask = np.random.permutation(range(len(X)))
X = X[mask]
y = y[mask].reshape(-1, 1)  # there are 3 different classes in y.

## Assign to training and testing sets
X_train = X[test_size:, :]
y_train = y[test_size:, :]
X_test = X[:test_size, :]
y_test = y[:test_size, :]

print("Shape of X_train:", X_train.shape, '\n', X_train[:5])
print("Shape of y_train", y_train.shape, '\n', y_train[:5])
print("Shape of y_test", y_test.shape, '\n', y_test[:5])


# Define the distance function
def euclid_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# print('euclidian distance:', euclid_distance(np.array([[1]]),np.array([[1],[0]])))

# Normalize the input
def normalize(x):
    for i in range(x.shape[1]):
        # print(i)
        # print(x[:, i])
        x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:,
                                                               i].min())  # normalize each column of x seperately. x_new = (x_old - x_old.min())/(x_old.max - x_old.min())

    return x


print("Normalization test", normalize(X_train[0:3]))


# Build the KNN model
def knn_predict(X_train, y_train, X_test, k=5):
    # 1. Normalization
    X_train = normalize(
        X_train)  # Here is a question, should we normalize X_train and X_test together to keep their relative position?
    X_test = normalize(X_test)
    print("Normalized X_train:\n", X_train)
    print("Normalized X_test:\n", X_test)

    predictions = np.empty(y_test.shape)    # initialize predictions

    for i in range(X_test.shape[0]):
        print("Current step:", i)
        distance = [euclid_distance(X_test[i], x_train) for x_train in
                    X_train]  # Compute the distance between the i-th point in X_test and all points in X_train.
        print("The distance between new point and points in X_train:", distance, type(distance[0]))
        neighbors_index = np.argsort(distance)[:k] # find the index of the k nearest points.
        print(neighbors_index)
        neighbors_y = np.array([y_train[j] for j in neighbors_index]) # find the feature of the k nearest points. type(neighbors_y) = list
        print(neighbors_y, type(neighbors_y))
        neighbors_y = neighbors_y.flatten()
        print(neighbors_y)
        pred_y = np.bincount(neighbors_y).argmax() # find the most frequent feature among the k nearest points, and this feature is what we predict for the new points.
        print(pred_y, type(pred_y))

        predictions[i] = np.array([pred_y]) # add the predicted y to final predictions


    print("The predictions for all points in X_test: ", predictions)

    return predictions


def accuracy(prediction, actual):
    return np.sum(prediction == actual) / len(prediction)


# prediction = knn_predict(X_train, y_train, X_test, k=9)
prediction = knn_predict(X_train, y_train, X_test)
print('Accuracy on test set: {:.2f}'.format(accuracy(prediction, y_test)))

