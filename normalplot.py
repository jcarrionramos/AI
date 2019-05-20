import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def Mean(data):
    return float(sum(data) / len(data))

def Std(data):
    mu = Mean(data)
    sum = 0
    for current in data:
        sum += (current - mu)**2

    variance = sum / len(data)
    return math.sqrt(variance)

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-20, 20, 0.001)

#
x = [4, 5, 7, 8, 8, 9, 10, 5, 2, 3, 5, 4, 8, 9]
plt.plot(x_axis, norm.pdf(x_axis, Mean(x), Std(x)), color = "red")

plt.plot(x_axis, norm.pdf(x_axis, np.mean(x), np.std(x)), color = "orange")

# Mean = 7, SD = 3.
plt.plot(x_axis, norm.pdf(x_axis,5,3), color = "blue")

plt.plot(x_axis, norm.pdf(x_axis,7,3), color = "green")

plt.show()
