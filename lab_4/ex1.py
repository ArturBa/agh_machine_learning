# fit the sigmoid curve and calculate decision boundary using given dataset

# a cheat sheet:
# in an optimization loop
# first calculate hypothesis for each datapoint x in X: h = 1 / (1 + exp(-theta0-theta1*x))
# then calculate crossentropy: -y*log(h) - (1-y)*log(1-h)
# and cost: sum(crossentropy) / len(x)
# next calculate derivatives for theta 0 and theta1 (similar to those in linear regression)
# theta0_deriv = sum(h - y) / len(y), theta1_deriv = sum((h-y)*X)
# and then update theta weights
# theta = theta - lr*theta_deriv

# check if cost is getting lower through iterations
# if not, try to modify the learning rate

# calculating decision boundary might look like this:
# theta[0] + theta[1]*x = 0
# theta[1]*x = -theta[0]
# x = -theta[0]/theta[1]

# the result might look like below

from matplotlib import pyplot as plt
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 25], dtype=np.float32)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1,  1,  1], dtype=np.float32)

theta = np.array([0, 0], dtype=np.float32)

# optimization loop

max_iter = 10000
eps = 0.00001
alpha = 0.05
prev_cost = 99


for i in range(max_iter):
    h_x = 1/(1 + np.exp(-theta[0] - theta[1]*X))

    crossentropy = -y*np.log(h_x+0.00001) - (1-y)*np.log(1-h_x+0.00001)
    cost = np.sum(crossentropy) / X.shape[0]

    theta_deriv = np.array([sum(h_x-y)/len(y), sum((h_x-y)*X)], dtype=np.float32)
    # theta0_deriv = sum(h - y) / len(y), theta1_deriv = sum((h-y)*X)

    # theta_derivs = sum((h_x-y) @ X) / X.shape[0]
    # theta_derivs.shape = [len(theta_derivs), 1]

    theta = theta - alpha*theta_deriv

    print("epoch ", str(i+1), ", cost ", cost)

    if np.abs(prev_cost - cost) < eps:
        break

    prev_cost = cost

border = -theta[0]/theta[1]
print(theta)

x_sim = np.linspace(1, 25)
# y_sim = np.log(x_sim.dot(theta[1])  + theta[0])
y_sim = 1/(1 + pow(np.e, - (x_sim.dot(theta[1]) + theta[0])))
plt.scatter(X, y)
plt.axvline(border, c='r')
plt.plot(x_sim, y_sim, c='g')
plt.show()


