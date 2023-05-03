import numpy as np 
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Reading in data
data = pd.read_csv("./input/diabetes.csv")
print(data)
X = np.zeros((768,8))
Y = np.zeros((768,1))
for index, row in data.iterrows():
    X[index][0] = row["Pregnancies"]
    X[index][1] = row["Glucose"]
    X[index][2] = row["BloodPressure"]
    X[index][3] = row["SkinThickness"]
    X[index][4] = row["Insulin"]
    X[index][5] = row["BMI"]
    X[index][6] = row["DiabetesPedigreeFunction"]
    X[index][7] = row["Age"]
    Y[index] = row["Outcome"]
#print("X: " + str(X))
#print("Y: " + str(Y))
#Splitting to training and testing samples
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.21875)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


#1a
X_train_0 = []
X_train_1 = []
for i in range(Y_train.shape[0]):
    if Y_train[i] == 0:
        X_train_0.append(X_train[i,:])
    else:
        X_train_1.append(X_train[i,:])
X_train_0 = np.array(X_train_0)
X_train_1 = np.array(X_train_1)
print("X_train_0 shape: " + str(X_train_0.shape))
print("X_train_1 shape: " + str(X_train_1.shape))

#1b
u_0 = np.zeros((8,1))
u_1 = np.zeros((8,1))
sigma_0 = np.zeros((8,1))
sigma_1 = np.zeros((8,1))
for i in range(X_train_0.shape[1]):
    u_0[i] = np.mean(X_train_0[:][i])
    u_1[i] = np.mean(X_train_1[:][i])
    sigma_0[i] = np.std(X_train_0[:][i])
    sigma_1[i] = np.std(X_train_1[:][i])
print("u_0: " + str(u_0))
print("u_1: " + str(u_1))
print("sigma_0: " + str(sigma_0))
print("sigma_1: " + str(sigma_1))

#1c
#i
p_0 = np.zeros((X_test.shape[0],1))
p_1 = np.zeros((X_test.shape[0],1))
for row in range(X_test.shape[0]):
    for i in range(X_test.shape[1]):
        temp = 1/(math.sqrt(2*math.pi) * sigma_0[i])
        e_temp = math.exp(-(X_test[row][i] - u_0[i])**2/(2*(sigma_0[i])**2))
        p_0[row] = p_0[row] + math.log(e_temp*temp)
#print("p_0: " + str(p_0))

for row in range(X_test.shape[0]):
    for i in range(X_test.shape[1]):
        temp = 1/(math.sqrt(2*math.pi) * sigma_1[i])
        e_temp = math.exp(-(X_test[row][i] - u_1[i])**2/(2*(sigma_1[i])**2))
        p_1[row] = p_1[row] + math.log(temp*e_temp)
#print("p_1: " + str(p_1))
p_0 += math.log(.65)
p_1 += math.log(.35)
#ii and iii
""""
ln_p_0 = np.zeros((X_train_0.shape[0],1))
ln_p_1 = np.zeros((X_train_1.shape[0],1))
for i in range(ln_p_0.shape[0]):
    ln_p_0[i] = math.log(p_0[i]) + math.log(.65)
for j in range(ln_p_1.shape[0]):
    ln_p_1[j] = math.log(p_1[j]) + math.log(.35)
"""
#iv
Y_pred = np.zeros((Y_test.shape[0],1))
for i in range(Y_pred.shape[0]):
    if p_0[i] > p_1[i]:
        Y_pred[i] = 0
    else:
        Y_pred[i] = 1
accuracy = metrics.accuracy_score(Y_test,Y_pred)
print("accuracy: " + str(accuracy))

#2
covariance_0 = np.cov(np.transpose(X_train_0))
covariance_1 = np.cov(np.transpose(X_train_1))
print("covariance_0 shape: " + str(covariance_0.shape))
print("covariance_1 shape: " + str(covariance_1.shape))
print("covariance_0: " + str(covariance_0))
print("covariance_1: " + str(covariance_0))


#TODO: 
l = 8
g_0 = np.zeros((X_test.shape[0],1))
det = np.linalg.det(covariance_0)
C_0 = -(l/2)*math.log(2*math.pi) - .5*math.log(det)
for i in range(X_test.shape[0]):
    X_temp = np.reshape(X_test[i][:],(8,1))
    #u_temp = np.reshape(u_0,(8,))
    first_term = np.transpose((X_temp - u_0))
    second_term = (X_temp - u_0)
    cov1_inv = np.linalg.pinv(covariance_0)
    temp1 = np.matmul(first_term,cov1_inv)
    temp2 = np.matmul(temp1,second_term)
    g_0[i] = -.5*temp2 + math.log(.65) + C_0
print(g_0.shape)
print("g_0: " + str(g_0))

g_1 = np.zeros((X_test.shape[0],1))
det2 = np.linalg.det(covariance_1)
C_1 = -(l/2)*math.log(2*math.pi) - .5*math.log(det2)
for i in range(X_test.shape[0]):
    X_temp = np.reshape(X_test[i][:],(8,1))
    u_temp = np.reshape(u_1,(8,))
    first_term = np.transpose((X_temp - u_1))
    second_term = (X_temp - u_1)
    cov2_inv = np.linalg.pinv(covariance_1)
    temp1 = np.matmul(first_term,cov2_inv)
    temp2 = np.matmul(temp1,second_term)
    g_1[i] = -.5*temp2 + math.log(.35) + C_1
print(g_1.shape)
print("g_1: " + str(g_1))

#iv
Y_pred2 = np.zeros((Y_test.shape[0],1))
for i in range(Y_pred2.shape[0]):
    if g_0[i] > g_1[i]:
        Y_pred2[i] = 0
    else:
        Y_pred2[i] = 1
acc = metrics.accuracy_score(Y_test,Y_pred2)
print("accuracy: " + str(acc))
