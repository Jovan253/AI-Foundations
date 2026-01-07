import numpy as np
import matplotlib.pyplot as plt
from model import LogisticRegression

X = np.array([
    [1], [2], [3], [4], [5], [6]
])
y = np.array([0, 0, 0, 1, 1, 1])


model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

X2 = np.array([
    [0], [4], [3], [1], [2], [6]
])

predictions = model.predict(X2)
print(predictions)

plt.plot(model.losses)
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Training Loss Over Time")
plt.show()
