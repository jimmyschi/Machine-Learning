from cProfile import label
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from gradientDescent import gradientDescent
from computeCost import computeCost
from normalEqn import normalEqn

data = pd.read_csv("./input/hw2_data1.csv")
print(data)
horsepower = data.iloc[:,0].copy()
price = data.iloc[:,1].copy()
plt.scatter(horsepower,price,color="red",marker="x")
plt.xlabel("Horse power of car in 100s")
plt.ylabel("Price in $1000s")
plt.savefig("./output/ps2-4-b.png")
plt.close()
#plt.show()

#feature matrix X
print(horsepower.shape)
X = np.ones((horsepower.shape[0],2))
for i in range(X.shape[0]):
    X[i][1] = horsepower[i]
y = price.to_numpy()
#print(X.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.10)
print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("y_test shape: " + str(y_test.shape))
theta, cost = gradientDescent(X_train,y_train,.3,500)
print(theta)

#4e
plt.plot(cost)
plt.savefig("./output/ps2-4-e.png")


#4f
print(theta.shape)
print(X_test.shape)
print(X_test[:,1])
y_test = theta[0] + theta[1]*X_test[:,1]
sum = 0
print(y_test.shape)
print(y_test.shape[0])
for i in range(y_test.shape[0]):
    sum = sum + (y_test[i] - y_train[i])**2
mse = sum/(2*y_test.shape[0])
print("cost: " + str(mse))

#4g
theta_norm = normalEqn(X_train,y_train)
y_test = theta_norm[0] + theta[1]*X_test[:,1]
sum2 = 0
for i in range(y_test.shape[0]):
    sum2 = sum2 + (y_test[i] - y_train[i])**2
mse2 = sum2/(2*y_test.shape[0])
print("cost: " + str(mse2))

#4h
plt.figure()
alpha = [.001, .003, .03, 3]
theta1, cost1 = gradientDescent(X_train,y_train,alpha[0],300)
#print("cost1: " + str(cost1))
plt.plot(cost1,label="a = .001")
plt.savefig("./output/ps2-4-h-1.png")
theta2, cost2 = gradientDescent(X_train,y_train,alpha[1],300)
#print("cost12: " + str(cost2))
plt.figure()
plt.plot(cost2,label="a = .003")
plt.savefig("./output/ps2-4-h-2.png")
theta3, cost3 = gradientDescent(X_train,y_train,alpha[2],300)
#print("cost3: " + str(cost3))
plt.figure()
plt.plot(cost3,label="a = .03")
plt.savefig("./output/ps2-4-h-3.png")
theta4, cost4 = gradientDescent(X_train,y_train,alpha[3],300)
#print("cost4: " + str(cost4))
plt.figure()
plt.plot(cost4,label="a = 3")
plt.savefig("./output/ps2-4-h-4.png")

