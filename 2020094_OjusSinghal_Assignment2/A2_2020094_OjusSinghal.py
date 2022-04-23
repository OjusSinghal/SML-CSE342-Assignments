import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


def plot_MLE_vs_N(data, N):
    plots = []

    for n in range(1, N + 1):
        mu_MLE = 0
        for i in range(n):
            mu_MLE += data[i]
        mu_MLE /= n
        plots.append(mu_MLE)
    
    plt.plot(plots)
    plt.show()

def get_MLE(data, N):
    mu_MLE = 0
    for i in range(N):
        mu_MLE += data[i]
    mu_MLE /= N
    return mu_MLE


N = 100
train_size = 50
test_size = N - train_size


mu1 = [0.5, 0.8]
mu2 = [0.9, 0.2]

x1 = np.array([bernoulli.rvs(mu1[0], size=N), bernoulli.rvs(mu1[1], size=N)])
x2 = np.array([bernoulli.rvs(mu2[0], size=N), bernoulli.rvs(mu2[1], size=N)])

plt.scatter(x1[0], x1[1], color='b', s=10)
plt.scatter(x2[0], x2[1], color='g', s=10)
plt.show()

x1 = np.array([[x1[0][i], x1[1][i]] for i in range(N)], dtype='float')
x2 = np.array([[x2[0][i], x2[1][i]] for i in range(N)], dtype='float')

plot_MLE_vs_N(x1, train_size)
plot_MLE_vs_N(x2, train_size)

mu1_MLE = get_MLE(x1, train_size)
mu2_MLE = get_MLE(x2, train_size)

d = 2

correct_classifications = 0
total = 0

for k in range(train_size, N):

    g1x = 0
    for j in range(d):
        g1x += x1[k][j] * np.log(mu1_MLE[j])
        g1x += (1 - x1[k][j]) * np.log(1 - mu1_MLE[j])

    g2x = 0
    for j in range(d):
        g2x += x1[k][j] * np.log(mu2_MLE[j])
        g2x += (1 - x1[k][j]) * np.log(1 - mu2_MLE[j])

    total += 1
    if g1x > g2x: correct_classifications += 1
    

for k in range(train_size, N):

    g1x = 0
    for j in range(d):
        g1x += x2[k][j] * np.log(mu1_MLE[j])
        g1x += (1 - x2[k][j]) * np.log(1 - mu1_MLE[j])

    g2x = 0
    for j in range(d):
        g2x += x2[k][j] * np.log(mu2_MLE[j])
        g2x += (1 - x2[k][j]) * np.log(1 - mu2_MLE[j])

    total += 1
    if g1x < g2x: correct_classifications += 1

# print(total)
print("\ncorrect classifications:", str(correct_classifications * 100 / total) + "%\n")

#########################################################################################
#########################################################################################
#########################################################################################


import numpy as np

X = np.array([[1, 7], [2, 5]], dtype='float')
mu_x = np.mean(X, axis=0)

X_C = X - mu_x
S = np.cov(np.transpose(X_C), bias=True)

W, v = np.linalg.eig(S)

idx = W.argsort()[::-1]   
W = W[idx]
v = -v[:,idx]
Y = np.matmul(np.transpose(v), np.transpose(X_C))

temp = (np.transpose(np.matmul(v, Y)) + mu_x) - X

MSE = 0.5 * np.linalg.det(np.matmul(np.transpose(temp), temp))
print("MSE = " + str(MSE))



#########################################################################################
#########################################################################################
#########################################################################################


import numpy as np
import matplotlib.pyplot as plt


N = 10000
D = 100

mean = np.random.uniform(0, 1, D)
cov = np.random.rand(D, D)
cov = np.dot(cov, np.transpose(cov))

X = np.random.multivariate_normal(mean, cov, N)
W, v = np.linalg.eig(cov)

mean = X.mean(axis=0)
X -= mean
cov = np.cov(np.transpose(X))

W, U = np.linalg.eig(cov)

idx = W.argsort()[::-1]   
W = W[idx]
U = -U[:,idx]

print("U:")
print(U)

Y = np.matmul(np.transpose(U), np.transpose(X))

temp = (np.transpose(np.matmul(U, Y)) + mean) - X

MSE = 0.5 * np.linalg.det(np.matmul(np.transpose(temp), temp))
print("MSE = " + str(MSE))

MSEs = np.zeros(D)
xs = np.arange(1, D + 1)

for i in range(1, D + 1):
    this_U = np.reshape(U[0 : i], (i , D))
    this_Y = np.matmul(this_U, np.transpose(X))
    temp = (np.transpose(np.matmul(np.transpose(this_U), this_Y)) + mean) - X

    MSEs[i - 1] = sum(np.square(temp)).mean(axis=0) / N
    print("MSE for", i, "principal component(s) = ", MSEs[i - 1])

plt.plot(xs, MSEs)
plt.xlabel("number of principal components")
plt.ylabel("MSE")
plt.show()