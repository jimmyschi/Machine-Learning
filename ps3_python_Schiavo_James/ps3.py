import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math


data_size = open("./input/ps3_template/input/hw3_data1.txt","r+")
size = len(data_size.readlines())
features = np.zeros((size,2))
y = np.zeros((size,1))
data_size.close()
data = open("./input/ps3_template/input/hw3_data1.txt")
count = 0
for lines in data: 
    line = lines.split(",")
    features[count][0] = line[0]
    features[count][1] = line[1]
    y[count] = line[2]
    count = count + 1
#print(features.shape)
#print(features)

X = np.ones((features.shape[0],features.shape[1] + 1))
for i in range(features.shape[0]):
    X[i][1] = features[i][0]
    X[i][2] = features[i][1]
print("X shape: " + str(X.shape))
print(X)
print("y shape: " + str(y.shape))
print(y)

colors = ['r' if yy==1 else 'b' for yy in y]
#print(colors)
plt.figure()
plt.scatter(X[:,1],X[:,2],c=colors)
plt.savefig("./output/ps3-1-b.png")

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.10)

from sigmoid import sigmoid
z = np.arange(start=-15,stop=16,step=1)
#print("z: " + str(z))
gz = sigmoid(z)
#print("gz: " + str(gz))
plt.figure()
plt.plot(z,gz)
plt.savefig("./output/ps3-1-c.png")

from scipy.optimize import fmin_bfgs
from costFunction import costFunction
from gradientDescent import gradientDescent
theta = np.zeros((X_train.shape[1],1))
theta_min = fmin_bfgs(x0=theta,f=costFunction,fprime=gradientDescent,args=(X_train,y_train),maxiter=400)
print("theta_min: " + str(theta_min))


from matplotlib import lines as mlines

colors = ['r' if yy==1 else 'b' for yy in y]
plt.figure()
#fig, ax = plt.subplots()
plt.scatter(X[:,1],X[:,2],c=colors)
X1 = np.arange(3,10,.1)
X2 = (-X1*theta_min[1] - theta_min[0])/theta_min[2]
#line = mlines.Line2D(X1,X2, color='red')
#transform = ax.transAxes
#line.set_transform(transform)
#ax.add_line(line)
plt.plot(X1,X2)
plt.savefig("./output/ps3-1-f.png")

#1i
y_pred = 1/(1 + math.exp(-1*(theta_min[0]*1 + theta_min[1]*50 + theta_min[2]*75)))
print(y_pred)

