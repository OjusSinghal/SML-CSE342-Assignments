import numpy as np
import matplotlib.pyplot as plt
import sys

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