import numpy as np
from predict import predict
from sigmoidGradient import sigmoidGradient
from nnCost import nnCost

def sGD(input_layer_size,hidden_layer_size,num_labels,X_train,y_train,lamb,alpha,MaxEpochs):
    y_k = np.zeros((y_train.shape[0],3))
    for i in range(y_k.shape[0]):
        if y_train[i] == 1:
            y_k[i] = [1,0,0]
        elif y_train[i] == 2:
            y_k[i] = [0,1,0]
        elif y_train[i] == 3:
            y_k[i] = [0,0,1]
    Theta1 = np.random.uniform(-.1,.1,(hidden_layer_size,X_train.shape[1]))
    #print("Theta1 shape: " + str(Theta1.shape))
    Theta2 = np.random.uniform(-.1,.1,(num_labels,hidden_layer_size + 1))
    #print("Theta2 shape: " + str(Theta2.shape))
    #delta_output = np.zeros((num_labels,hidden_layer_size + 1))
    #delta_hidden = np.zeros((hidden_layer_size,X_train.shape[1]))
    D_1 = np.zeros((hidden_layer_size,X_train.shape[1]))
    D_2 = np.zeros((num_labels,hidden_layer_size + 1))
    c = []
    for epoch in range(MaxEpochs):
        for m in range(X_train.shape[0]):
            p,h_x = predict(Theta1,Theta2,X_train)
            a_1 = X_train[m,:]
            z_2 = np.dot(Theta1,a_1)
            z_2_new = np.ones((z_2.shape[0] + 1,1))
            for i in range(1,z_2_new.shape[0]):
                z_2_new[i] = z_2[i -1]
            #print("z_2 shape: " + str(z_2.shape))
            a_2 = 1.0 / (1.0 + np.exp(-z_2))
            a_2_new = np.ones((a_2.shape[0] + 1,1))
            for i in range(1,a_2_new.shape[0]):
                a_2_new[i] = a_2[i -1]
            delta_output = h_x[m] - y_train[m]
            #print("delta_output: " + str(delta_output))
            temp = np.dot(Theta2.T,delta_output)
            z_output = sigmoidGradient(z_2_new)
            #print("temp shape: " + str(temp.shape))
            #print("z_output shape: " + str(z_output.shape))
            delta_hidden = temp*z_output
            delta_hidden = np.delete(delta_hidden,0,0)
            #print("delta_hidden shape: " + str(delta_hidden.shape))
            delta_hidden = delta_hidden.reshape(8,1)
            a_1 = a_1.reshape(1,5)
            delta_1 = np.dot(delta_hidden,a_1)
            delta_output = delta_output.reshape(3,1)
            a_2_new = a_2_new.reshape(1,9)
            delta_2 = np.dot(delta_output,a_2_new)
            for j in range(hidden_layer_size):
                for i in range(X_train.shape[1]):
                    if i != 0:
                        D_1[j][i] = delta_1[j][i] + lamb*Theta1[j][i]
                    else:
                        D_1[j][i] = delta_1[j][i]
            #print("D_1: " + str(D_1))
            for k in range(num_labels):
                for j in range(hidden_layer_size + 1):
                    if k != 0:
                        D_2[k][j] = delta_2[k][j] + lamb*Theta2[k][j]
                    else:
                        D_2[k][j] = delta_2[k][j]
            #print("D_2: " + str(D_2))
            for j in range(hidden_layer_size):
                for i in range(X_train.shape[1]):
                    Theta1[j][i] = Theta1[j][i] - alpha*D_1[j][i]
            for k in range(num_labels):
                for j in range(hidden_layer_size + 1):
                    Theta2[k][j] = Theta2[k][j] - alpha*D_2[k][j]
        #print("Theta1: " + str(Theta1))
        #print("Theta2: " + str(Theta2))
        cost = nnCost(Theta1,Theta2,X_train,y_train,3,lamb)
        c.append(cost)
        #print("epoch: " + str(epoch))
        #print("cost: " + str(cost))
    return Theta1,Theta2, c
