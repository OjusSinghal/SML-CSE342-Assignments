import numpy as np
import matplotlib.pyplot as plt
import sys

# Q1

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


# Q3

def cauchy(z, a, b):
    temp = b * np.pi * (1 + ((z - a) / b) ** 2)
    if 0 in temp:
        print("cauchy function divide by zero error")
        exit(0)
    return 1 / temp


n = 1000

##########################################################################
##################### P(w1|x)

left = -20
right = 20

a1 = 3
a2 = 5
b = 1

x3 = np.linspace(left, right, n)
y3 = cauchy(x3, a1, b) / (cauchy(x3, a1, b) + cauchy(x3, a2, b))
# y1 = cauchy(x3, a1, b)
# y2 = cauchy(x3, a2, b)



##########################################################################
##################### Plotting


fig, ax = plt.subplots()
# ax.plot(x3, y1, 'b', label="P(x|w1)")
# ax.plot(x3, y2, 'g', label="P(x|w2)")
ax.plot(x3, y3, 'r', label="P(w1|x)")
plt.legend(loc="upper left")
plt.xlabel("x")


plt.ylim([0, 1])
plt.xlim([left, right])
plt.show()