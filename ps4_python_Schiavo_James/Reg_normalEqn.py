from cProfile import label
import numpy as np 
import scipy.io as sio
from matplotlib import pyplot as plt
from computeCost import computeCost

def Reg_normalEqn(X_train,y_train,lamb):
    X_t = np.transpose(X_train)
    X_t_X = np.matmul(X_t,X_train)
    #print(" X_t_x: " + str(X_t_X))
    diag = np.zeros((X_train.shape[1],X_train.shape[1]))
    for i in range(1,diag.shape[0]):
        for j in range(1,diag.shape[1]):
            if i == j:
                diag[i][j] = 1
    #print("diag: " + str(diag))
    inv_mat = X_t_X + lamb*diag
    inv = np.linalg.pinv(inv_mat)
    X_t_y = np.matmul(X_t,y_train)
    theta = np.matmul(inv, X_t_y)
    return theta

""""
data2 = sio.loadmat("./input/hw4_data1.mat")
X_data = np.array(data2["X_data"])
one = np.ones((X_data.shape[0],1))
X = np.concatenate((one,X_data),axis=1)
y = np.array(data2["y"])

from sklearn.model_selection import train_test_split
from computeCost import computeCost
lamb = [0.0,0.001,0.003,0.005,0.007,0.009,0.012,0.017]
training_error = np.zeros((20,8))
testing_error = np.zeros((20,8))
for i in range(20):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.12)
    for j in range(len(lamb)):
        theta = Reg_normalEqn(X_train,y_train,lamb[j])
        training_error[i][j] = computeCost(X_train,y_train,theta)
        testing_error[i][j] = computeCost(X_test,y_test,theta)
train_line = np.average(training_error,axis=0)
test_line = np.average(testing_error,axis=0)
plt.figure()
plt.plot(lamb,train_line,color='r',label="training error")
plt.plot(lamb,test_line,color='b',label="testing error")
plt.savefig("./output/ps4-1-a.png")

"""


