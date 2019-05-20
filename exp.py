import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import expon

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
x_axis = np.arange(6.5, 10, 0.1)

x = [4, 5, 7, 8, 8, 9, 10, 5, 2, 3, 5, 4, 8, 9]

plt.plot(x_axis, expon.pdf(x_axis, Mean(x)), color="blue")
plt.plot(x_axis, expon.pdf(x_axis, np.mean(x)), color = "green")

plt.show()
