## K-Means

The cost we use to evaluate the algorithm is the mean distance 
between data points and their cluster centroid. That is kind of wired.
Why don't we use the correct classification of the points to see 
how many points are predicted correctly?

Answer: K-Means method is an unsupervised algorithm, so basically we do
not have the correct classification of our data points. Instead, we 
use Elbow method to determine the number of clusters in the dataset.

The use of **scipy.spatial.distance** make the programing much easier.
It computes the distance between each pair of elements in two collections.

``` Python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Normalize X
X = 

class kmeans:
    def __init__(self, X, k=3, iters=1000):
        self.X       = 
        self.k       = 
        self.iters   = 
        self.m       = 
        self.n       = 
        self.centers = 
        self.minIdx  = 
        self.costVal = 

    def fit(self):
        for _ in range(self.iters):

        return self.minIdx

    def costFcn(self):
        for i in range(self.k):
            centers = self.centers[i].reshape((1, self.n))
            cluster_distance = distance.cdist(self.X[self.minIdx==i], centers)
        return self.costVal

y_var = []
for i in range(1, 6):
    myKM = kmeans(X, k=i)
    myKM.fit()
    y_var.append(myKM.costFcn())
x_axis = [x for x in range(1, 6)]
plt.plot(x_axis, y_var)
plt.show()
```
