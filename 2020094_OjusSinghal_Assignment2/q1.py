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