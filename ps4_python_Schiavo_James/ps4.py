import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

#1
from Reg_normalEqn import Reg_normalEqn
data1 = sio.loadmat("./input/hw4_data1.mat")
X_data = np.array(data1["X_data"])
one = np.ones((X_data.shape[0],1))
X = np.concatenate((one,X_data),axis=1)
print("X.shape: " + str(X.shape))
y = np.array(data1["y"])

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

#2
from sklearn.neighbors import KNeighborsClassifier
data2 = sio.loadmat("./input/hw4_data2.mat")
X1 = np.array(data2["X1"])
X2 = np.array(data2["X2"])
X3 = np.array(data2["X3"])
X4 = np.array(data2["X4"])
X5 = np.array(data2["X5"])
y1 = np.array(data2["y1"])
y2 = np.array(data2["y2"])
y3 = np.array(data2["y3"])
y4 = np.array(data2["y4"])
y5 = np.array(data2["y5"])

X_train1 = np.concatenate((X1,X2,X3,X4),axis=0)
y_train1 = np.concatenate((y1,y2,y3,y4),axis=0)
X_test1 = X5
y_test1 = y5

X_train2 = np.concatenate((X1,X2,X3,X5),axis=0)
y_train2 = np.concatenate((y1,y2,y3,y5),axis=0)
X_test2 = X4
y_test2 = y4 

X_train3 = np.concatenate((X1,X2,X4,X5),axis=0)
y_train3 = np.concatenate((y1,y2,y4,y5),axis=0)
X_test3 = X3
y_test3 = y3

X_train4 = np.concatenate((X1,X3,X4,X5),axis=0)
y_train4 = np.concatenate((y1,y3,y4,y5),axis=0)
X_test4 = X2
y_test4 = y2

X_train5 = np.concatenate((X2,X3,X4,X5),axis=0)
y_train5 = np.concatenate((y2,y3,y4,y5),axis=0)
X_test5 = X1
y_test5 = y1

K = [1,3,5,7,9,11,13,15]
acc = []
from sklearn import metrics
for i in range(len(K)):
    knn = KNeighborsClassifier(K[i])
    knn.fit(X_train1,y_train1)
    y_pred1 = knn.predict(X_test1)
    acc1 = metrics.accuracy_score(y_test1,y_pred1)
    knn.fit(X_train2,y_train2)
    y_pred2 = knn.predict(X_test2)
    acc2 = metrics.accuracy_score(y_test2,y_pred2)
    knn.fit(X_train3,y_train3)
    y_pred3 = knn.predict(X_test3)
    acc3 = metrics.accuracy_score(y_test3,y_pred3)
    knn.fit(X_train4,y_train4)
    y_pred4 = knn.predict(X_test4)
    acc4 = metrics.accuracy_score(y_test4,y_pred4)
    knn.fit(X_train5,y_train5)
    y_pred5 = knn.predict(X_test5)
    acc5 = metrics.accuracy_score(y_test5,y_pred5)
    acc.append((acc1 + acc2 + acc3 + acc4 + acc5)/5)
plt.figure()
plt.plot(K,acc)
plt.savefig("./output/ps4-2-a.png")

#3
from logReg_multi import logReg_multi
data3 = sio.loadmat("./input/hw4_data3.mat")
X_train = np.array(data3["X_train"])
X_test = np.array(data3["X_test"])
y_train = np.array(data3["y_train"])
y_test = np.array(data3["y_test"])
y_predict1 = logReg_multi(X_train,y_train,X_test)
testing_accuracy = metrics.accuracy_score(y_test,y_predict1)
y_predict2 = logReg_multi(X_train,y_train,X_train)
training_accuracy = metrics.accuracy_score(y_train,y_predict2)
print("training accuracy: " + str(training_accuracy))
print("testing accuracy: " + str(testing_accuracy))


