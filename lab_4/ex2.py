# generalize optimization code for X being a matrix, where its rows are features and columns are examples
# code should work independently from number of features and number of examples
# use matrix multiplication (np.matmul or @)
# plot decision boundary on a plot x2(x1)
# calculating decision boundary might look like this:
# theta0 + theta1*x1 + theta2*x2 = 0
# theta2*x2 = -theta0 - theta1*x1
# x2 = -theta0/theta2 - theta1/theta2 * x1


from matplotlib import pyplot as plt
import numpy as np

X = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1], # bias' 'variables' already appended to X
              [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 25],
              [13, 9, 8, 6, 4, 2, 1, 0, 3,  4,  2]], dtype=np.float32)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1,  1,  1], dtype=np.float32)

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

X_positive = X[:, y[:] == 1]
X_negative = X[:, y[:] == 0]

plt.plot(X_positive[2, :], X_positive[1, :], '+')
plt.plot(X_negative[2, :], X_negative[1, :], 'o')
plt.plot(x2, x1, '-')
plt.show()
