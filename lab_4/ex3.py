# using real data, optimize classifier to predict given values

# split dataset into a training set and a test set
# train model on the training set
# calculate TP, FP, TN, FN on test set
# calculate sensitivity, specificity, positive predictively and negative predictively


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('./data.txt')
data = data.values

# lets try to put data into X and y.
# we want X to be n+1 by m (n - number of features, m - number of examples)
# and y to be 1 by m

# data is 99x3, 99 - examples, first two columns - features, last column - labels

X = np.ones((data.shape[1], data.shape[0]))  # create X of size 3x99
X[1:3, :] = data[:, 0:2].T  # fill X's second and third row with features, leave first row with ones

y = data[:, 2:3].T  # copy third column to y as row

# we may want to normalize the dataset in order to converge faster
X[1, :] = (X[1, :] - np.std(X[1, :])) / np.mean(X[1, :])
X[2, :] = (X[2, :] - np.std(X[2, :])) / np.mean(X[2, :])

# rest is the same as in 2.
theta = np.zeros((X.shape[0], 1))

max_iter = 10000
eps = 0.00001
alpha = 0.05
prev_cost = 99

for i in range(max_iter):
    h_x = 1 / (1 + np.exp(-theta.T@X))

    crossentropy = -y*np.log(h_x+0.00001) - (1-y)*np.log(1-h_x+0.00001)
    [cost] = np.sum(crossentropy, axis=1) / X.shape[1]

    theta_derivs = sum((h_x-y) @ X.T) / X.shape[1]
    theta_derivs.shape = [len(theta_derivs), 1]

    theta = theta - alpha*theta_derivs

    print("epoch ", str(i+1), ", cost ", cost)

    if np.abs(prev_cost - cost) < eps:
        break

    prev_cost = cost

print(theta)

x1 = np.linspace(np.min(X[1, :]), np.max(X[1, :]), 100)
x2 = -theta[0, 0]/theta[2, 0] - theta[1, 0]/theta[2, 0] * x1

X_positive = X[:, y[0, :] == 1]
X_negative = X[:, y[0, :] == 0]

plt.plot(X_positive[2, :], X_positive[1, :], '+')
plt.plot(X_negative[2, :], X_negative[1, :], 'o')
plt.plot(x2, x1, '-')
plt.show()