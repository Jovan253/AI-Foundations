Logistic Regression is used in Classification problems, used to figure out a probability rather than a numerical value.

## The Model

The model remains the same.

$z = Wx + b$

And what we're looking to do is to transform this to get probabilities for classes of 0 or 1. 


## Sigmoid Function

We transform z into a probability

$\hat{y} = \sigma{(z)} = \frac{1}{1+e^{-z}}$

If predicted value is close to 0 then class is 0, otherwise 1. (The Decision Rule)


## Loss Function

Log loss is used in classification as it's ideal for probabilistic outputs.

$J(w,b) = -\frac{1}{n}\Sigma[ylog(\hat{y}) + (1-y)log(1-\hat{y})]$

(This heavily penalizes confident wrong predictions)
