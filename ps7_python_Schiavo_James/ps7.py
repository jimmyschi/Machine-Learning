import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

from predict import predict
from nnCost import nnCost
from sigmoidGradient import sigmoidGradient
from sGD import sGD

from sklearn import metrics
from sklearn.model_selection import train_test_split

#0
data = sio.loadmat("./input/HW7_Data.mat")
weights = sio.loadmat("./input/HW7_weights_2.mat")
#print("data: " + str(data))
#print("weights: " + str(weights))
X = np.array(data["X"])
print("X: " + str(X))
y = np.array(data["y"])
print("y: " + str(y))
Theta1 = np.array(weights["Theta1"])
print("Theta1: " + str(Theta1))
Theta2 = np.array(weights["Theta2"])
print("Theta2: " + str(Theta2))
X_matrix = np.ones((X.shape[0],X.shape[1] + 1))
for i in range(1,X_matrix.shape[1]):
    X_matrix[:,i] = X[:,i-1]
print("X_matrix: " + str(X_matrix))

#1
p, h_x = predict(Theta1,Theta2,X_matrix)
p = p.reshape(-1,1)
h_x = h_x.reshape(h_x.shape[0],h_x.shape[1])
print("p shape: " + str(p.shape))
print("p: " + str(p))
print("h_x shape: " + str(h_x.shape))
print("h_x: " + str(h_x))
#1b
accuracy = metrics.accuracy_score(y,p)
print("accuracy: " + str(accuracy))

#2
K = 3
lamb = 0
J = nnCost(Theta1,Theta2,X_matrix,y,K,lamb)
print("J: " + str(J))

lamb = 1
J = nnCost(Theta1,Theta2,X_matrix,y,K,lamb)
print("J: " + str(J))

lamb = 2
J = nnCost(Theta1,Theta2,X_matrix,y,K,lamb)
print("J: " + str(J))

#3
z = [-10,0,10]
g_prime = sigmoidGradient(z)
print("g_prime: "  + str(g_prime))

#4
Theta1_new,Theta2_new, c = sGD(input_layer_size=X_matrix.shape[0],hidden_layer_size=8,
num_labels=3,X_train=X_matrix,y_train=y,lamb=0,alpha=.001,MaxEpochs=100)
plt.plot(c)
plt.savefig("./output/ps7-4-e-1.png")

#5
X_train,X_test,y_train,y_test = train_test_split(X_matrix,y,test_size=.15)
Theta1_new,Theta2_new,c = sGD(input_layer_size=X_train.shape[0],hidden_layer_size=8,
num_labels=3,X_train=X_train,y_train=y_train,lamb=0,alpha=.001,MaxEpochs=50)

p_train,h_x = predict(Theta1_new,Theta2_new,X_train)
training_accuracy = metrics.accuracy_score(y_train,p_train)
print("training accuracy: " + str(training_accuracy))
p_test,h_x = predict(Theta1_new,Theta2_new,X_test)
testing_accuracy = metrics.accuracy_score(y_test,p_test)
print("testing accuracy: " + str(testing_accuracy))

Theta1_new,Theta2_new,c = sGD(input_layer_size=X_train.shape[0],hidden_layer_size=8,
num_labels=3,X_train=X_train,y_train=y_train,lamb=.01,alpha=.001,MaxEpochs=50)

p_train,h_x = predict(Theta1_new,Theta2_new,X_train)
training_accuracy = metrics.accuracy_score(y_train,p_train)
print("training accuracy: " + str(training_accuracy))
p_test,h_x = predict(Theta1_new,Theta2_new,X_test)
testing_accuracy = metrics.accuracy_score(y_test,p_test)
print("testing accuracy: " + str(testing_accuracy))

Theta1_new,Theta2_new,c = sGD(input_layer_size=X_train.shape[0],hidden_layer_size=8,
num_labels=3,X_train=X_train,y_train=y_train,lamb=.1,alpha=.001,MaxEpochs=50)

p_train,h_x = predict(Theta1_new,Theta2_new,X_train)
training_accuracy = metrics.accuracy_score(y_train,p_train)
print("training accuracy: " + str(training_accuracy))
p_test,h_x = predict(Theta1_new,Theta2_new,X_test)
testing_accuracy = metrics.accuracy_score(y_test,p_test)
print("testing accuracy: " + str(testing_accuracy))

Theta1_new,Theta2_new,c = sGD(input_layer_size=X_train.shape[0],hidden_layer_size=8,
num_labels=3,X_train=X_train,y_train=y_train,lamb=1,alpha=.001,MaxEpochs=50)

p_train,h_x = predict(Theta1_new,Theta2_new,X_train)
training_accuracy = metrics.accuracy_score(y_train,p_train)
print("training accuracy: " + str(training_accuracy))
p_test,h_x = predict(Theta1_new,Theta2_new,X_test)
testing_accuracy = metrics.accuracy_score(y_test,p_test)
print("testing accuracy: " + str(testing_accuracy))

Theta1_new,Theta2_new,c = sGD(input_layer_size=X_train.shape[0],hidden_layer_size=8,
num_labels=3,X_train=X_train,y_train=y_train,lamb=0,alpha=.001,MaxEpochs=100)

p_train,h_x = predict(Theta1_new,Theta2_new,X_train)
training_accuracy = metrics.accuracy_score(y_train,p_train)
print("training accuracy: " + str(training_accuracy))
p_test,h_x = predict(Theta1_new,Theta2_new,X_test)
testing_accuracy = metrics.accuracy_score(y_test,p_test)
print("testing accuracy: " + str(testing_accuracy))

Theta1_new,Theta2_new,c = sGD(input_layer_size=X_train.shape[0],hidden_layer_size=8,
num_labels=3,X_train=X_train,y_train=y_train,lamb=.01,alpha=.001,MaxEpochs=100)

p_train,h_x = predict(Theta1_new,Theta2_new,X_train)
training_accuracy = metrics.accuracy_score(y_train,p_train)
print("training accuracy: " + str(training_accuracy))
p_test,h_x = predict(Theta1_new,Theta2_new,X_test)
testing_accuracy = metrics.accuracy_score(y_test,p_test)
print("testing accuracy: " + str(testing_accuracy))

Theta1_new,Theta2_new,c = sGD(input_layer_size=X_train.shape[0],hidden_layer_size=8,
num_labels=3,X_train=X_train,y_train=y_train,lamb=.1,alpha=.001,MaxEpochs=100)

p_train,h_x = predict(Theta1_new,Theta2_new,X_train)
training_accuracy = metrics.accuracy_score(y_train,p_train)
print("training accuracy: " + str(training_accuracy))
p_test,h_x = predict(Theta1_new,Theta2_new,X_test)
testing_accuracy = metrics.accuracy_score(y_test,p_test)
print("testing accuracy: " + str(testing_accuracy))

Theta1_new,Theta2_new,c = sGD(input_layer_size=X_train.shape[0],hidden_layer_size=8,
num_labels=3,X_train=X_train,y_train=y_train,lamb=1,alpha=.001,MaxEpochs=100)

p_train,h_x = predict(Theta1_new,Theta2_new,X_train)
training_accuracy = metrics.accuracy_score(y_train,p_train)
print("training accuracy: " + str(training_accuracy))
p_test,h_x = predict(Theta1_new,Theta2_new,X_test)
testing_accuracy = metrics.accuracy_score(y_test,p_test)
print("testing accuracy: " + str(testing_accuracy))

