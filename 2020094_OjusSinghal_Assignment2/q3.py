import numpy as np

X = np.array([[1, 7], [2, 5]], dtype='float')
print(X.shape)
mu_x = np.mean(X, axis=0)

X_C = X - mu_x
S = np.cov(np.transpose(X_C), bias=True)

W, v = np.linalg.eig(S)
idx = W.argsort()[::-1]   
W = W[idx]
v = v[:,idx]
Y = np.matmul(np.transpose(v), np.transpose(X_C))

temp = (np.transpose(np.matmul(v, Y)) + mu_x) - X


MSE = 0.5 * np.linalg.det(np.matmul(np.transpose(temp), temp))
print("MSE = " + str(MSE))