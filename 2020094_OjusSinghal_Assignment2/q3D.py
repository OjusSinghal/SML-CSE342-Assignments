import numpy as np
import matplotlib.pyplot as plt


N = 200
D = 20

mean = np.random.uniform(0, 1, D)
cov = np.random.rand(D, D)
cov = np.dot(cov, np.transpose(cov))

X = np.random.multivariate_normal(mean, cov, N)

mean = X.mean(axis=0)
X -= mean
X_C = X - mean
cov = np.cov(np.transpose(X_C))

W, U = np.linalg.eig(cov)

idx = W.argsort()[::-1]
W = W[idx]
U = -U[:, idx]

print("U:")
print(U)

Y = np.matmul(np.transpose(U), np.transpose(X_C))
print(Y)
temp = (np.transpose(np.matmul(U, Y)) + mean) - X

MSE = sum(np.square(temp)).mean(axis=0) / N
print("MSE = " + str(MSE))

MSEs = np.zeros(D)
xs = np.arange(1, D + 1)

for i in range(1, D + 1):
    this_U = np.reshape(U[0:i], (i, D))
    this_Y = np.matmul(this_U, np.transpose(X_C))
    temp = (np.transpose(np.matmul(np.transpose(this_U), this_Y)) + mean) - X

    MSEs[i - 1] = sum(np.square(temp)).mean(axis=0) / N
    print("MSE for", i, "principal component(s) = ", MSEs[i - 1])

plt.plot(xs, MSEs)
plt.xlabel("number of principal components")
plt.ylabel("MSE")
plt.show()
