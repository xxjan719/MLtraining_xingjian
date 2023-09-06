## Artificial Neural Network  
  
### Structure of ANN (One hidden layer)  
In this tutorial, we will employ ANN (with only one hidden layer) to solve the classification problem.  
We will first introduce the Digits dataset and One-Hot Encoding followed by the feed-forward and back-propagation.  
  
#### 1. Digits Dataset  
Digits Dataset is made up of 1797 8-by-8 images. Randomly split the dataset into training set and test set.  
* `X_train.shape` = (1347, 64)  
* `X_test.shape` = (450, 64)  
* `y_train.shape` = (1347, 1)  
* `y_test.shape` = (4501, 1)  
Here, 1347 is the number of training samples, 450 is the number of test samples, 64 is the feature size, and 10 is the number of classes.  
  
#### 2. One-Hot Encoding  
One-Hot Encoding can represent a  vector [0, 1, 2] as [100, 010, 001].  
  
#### 3. Feed-Forward  
We will construct a deep neural network with only one hidden layer. The input layer has 64 neurons. The $i$-th sample $x^i$ goes:  
1. In 1st hidden layer:  
$$z_1^i = W_1 x^i + b_1$$  
where $W_1 \in \mathbb{R}^{N_1\times 64}$, $b_1\in \mathbb{R}^{N_1\times 1}$, $z_1\in\mathbb{R}^{N_1\times 1}$.  
In each neuron, an activation function is added i.e. `relu` or `tanh`.  
$$f_1^i = tanh(z_1^i),$$  
where $f_1^i\in\mathbb{R}^{N_1\times 1}$.  
2. In the output layer:  
$$z_2^i = W_2 f_1^i + b_2$$  
where $W_2\in\mathbb{R}^{10\times N_1}$ and $b_2\in\mathbb{R}^{10\times 1}$.  
3. Use softmax function to get probability for each class:  
$$\hat{y}_i=softmax(z_2^i),$$  
where $softmax(z)_j = \frac{e^{z_j}}{\sum_ke^{z_j}}$  
  
#### 4. Back-Propagation  
1. Loss function (cross-entropy loss)  
$$L = -\sum_{i}y_ilog(\hat{y}_i)$$  
2. Derive the derivatives of $\frac{\partial L}{\partial W_2}$ $\frac{\partial L}{\partial b_2}$ $\frac{\partial L}{\partial W_1}$ $\frac{\partial L}{\partial b_1}$

## Structure of NN for this task
### Feed-Forward 
For a singal $x_i \in X\_train$, $x_i \in \mathbb{R}^{1\times 64}$. $W_1\in \mathbb{R}^{64\times 64}$ and $b_1 \in \mathbb{R}^{1\times 64}$. Then in the first hidden layer,
$$z_1 = x_i \cdot W_1+b_1 $$

$$f_1 = tanh(z_1)$$ so $f_1$ should have the same dimension with $z_1$, $f_1\in \mathbb{R}^{1\times 64}$.

For the output layer:

$$z_2 = f_1 \cdot W_2 +b_2$$

where $W_2\in \mathbb{R}^{64\times 10}$, $b_2 \in \mathbb{R}^{1\times 10}$, $z_2 \in \mathbb{R}^{1 \times 10}$.

Use softmax function to get probability for each class:
$$\hat{y}_i = softmax(z_2)$$

Fianlly, the Loss function is $$L = -\sum_{i}y_ilog(\hat{y}_i)$$ 

#### Feed-Forward in matrix computation form
$X\in \mathbb{R}^{1347\times 64}$, $W_1\in \mathbb{R}^{64\times 64}$, $b_1 \in \mathbb{R}^{1347\times 64}$, $Z_1 \in \mathbb{R}^{1347\times 64}$
The first hidden layer:
$$Z_1 = X\cdot W_1 + b_1$$
$$F_1 = tanh(Z_1)$$

The output layer:
$$Z_2 = F_1 \cdot W_2 +b_2$$
where $F_1\in \mathbb{R}^{1347\times 64}$, $W_2\in \mathbb{R}^{64\times 10}$, $b_2 \in \mathbb{R}^{1347\times 10}$, so $Z_2\in \mathbb{R}^{1347\times 64}$

Next, apply softmax function to every row of matrix $Z_2$,
$$\hat{Y} = softmax(Z_2)$$

