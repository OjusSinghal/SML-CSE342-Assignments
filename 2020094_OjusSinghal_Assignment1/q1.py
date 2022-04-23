import numpy as np
import matplotlib.pyplot as plt
import sys

def gaus(z, mean, var):
    temp = ((2 * np.pi * var) ** 0.5) * np.exp(((z - mean) ** 2) / (2 * var))
    return 1 / temp
n = 1000

##########################################################################
##################### P(x|w1)
mean1 = 2
var1 = 1

x1 = np.linspace(mean1 - 5 * var1, mean1 + 5 * var1, n)
y1 = gaus(x1, mean1, var1)

##########################################################################
##################### P(x|w2)
mean2 = 5
var2 = 1

x2 = np.linspace(mean2 - 5 * var2, mean2 + 5 * var2, n)
y2 = gaus(x2, mean2, var2)

##########################################################################
##################### P(x|w1) / P(x|w2)

left = min(mean1 - (3 * var1), mean2 - (3 * var2))
right = max(mean1 + (3 * var1), mean2 + (3 * var2))

x3 = np.linspace(left, right, n)
y3 = gaus(x3, mean1, var1) / gaus(x3, mean2, var2)



##########################################################################
##################### Plotting


fig, ax = plt.subplots()
ax.plot(x1, y1, 'b', label="P(x|w1)")
ax.plot(x2, y2, 'g', label="P(x|w2)")
ax.plot(x3, y3, 'r', label="P(x|w1) / P(x|w2)")
plt.legend(loc="upper left")
plt.xlabel("x")


plt.ylim([0, 1])
plt.xlim([-2, 8])
plt.show()