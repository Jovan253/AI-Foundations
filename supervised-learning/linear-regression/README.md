# Linear Regression

This module implements linear regression using gradient descent without relying on machine learning libraries

Linear Regression models the relationship between input features and a continuous target by fitting a linear function to the data.

### The Model

The model can be defined as $\hat{y} = Xw + b$

Our matrix X is the feature matrix, each row is a sample and each column represents a feature

$\hat{y}$ is the predicted output

Our weights, w, shows how important each feature is
And Bias shifts our line up and down, b

The true labels y are a vector of true values corresponding to each sample in X

The goal is to find the w and b so predictions are as close to real values.


### Loss Function

The Loss function is used to calculate the error, how far are we away from the real values.

One such popular function is Mean Squared Error:

$J(w, b) = \frac{1}{n} Σ (y - ŷ)²$

- n is the number of samples
- y is the true value
- $\hat{y}$ is the predicted value

The lower the loss the better the models performance


### Gradient Descent

Gradient Descent is used to minimize the loss by updating the weights and biases iteratively.

The intuition here is that we should move our parameters in the direction of steepest descent.

Hence from this we should update our weights and biases by our gradient descent.

Taking the partial derivatives of w and b from MSE we get:
$$
$$

Then we update the parameters providing a learning rate to control the step size.

We do this over many iterations, allowing our step to move closer to the minimum