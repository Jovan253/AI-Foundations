import numpy as np
import matplotlib.pyplot as plt
from model import LinearRegression
import sys
sys.path.append('../')
from mse import mean_squared_error

X = np.array([
    [1.0, 1],
    [2.0, 1],
    [3.0, 2],
    [4.0, 2],
    [5.0, 3],
    [6.0, 3]
])

# y depends on both features
y = np.array([3, 5, 8, 10, 13, 15])

model = LinearRegression()
model.fit(X,y)

predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

print("MSE:", mse)

# Plot Losses
plt.plot(model.losses)
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss Over Time")
plt.show()