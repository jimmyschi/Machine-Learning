import imp
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

data = pd.read_csv("./input/ps3_template/input/hw3_data2.csv")
pop = data.iloc[:,0].copy()
profit = data.iloc[:,1].copy()

print("pop: " + str(pop))
print("profit: " + str(profit))

from normalEqn import normalEqn
X = np.ones((pop.shape[0],3))
for i in range(pop.shape[0]):
    X[i][1] = pop[i]
    X[i][2] = pop[i]**2
print(X)
y = np.zeros((profit.shape[0],1))
for i in range(y.shape[0]):
    y[i] = profit[i]
print(profit.shape)
theta = normalEqn(X,y)
print("Theta: " + str(theta))

plt.figure()
plt.scatter(X[:,1],y)
y_pred = np.zeros((X.shape[0],1))
for i in range(y_pred.shape[0]):
    y_pred[i] = theta[0]*X[i][0] + theta[1]*X[i][1] + theta[2]*X[i][2]
plt.plot(X[:,1],y_pred,color='r')
plt.savefig("./output/ps3-2-b.png")