import numpy as np

def predict(Theta1,Theta2,X):
    h_x_output = []
    p_output = []
    for i in range(X.shape[0]):
        a_1 = X[i,:]
        z_2 = np.dot(Theta1,a_1)
        a_2 = 1.0 / (1.0 + np.exp(-z_2))
        a_2_new = np.ones((a_2.shape[0] + 1,1))
        for i in range(1,a_2_new.shape[0]):
            a_2_new[i] = a_2[i -1]
        z_3 = np.dot(Theta2,a_2_new)
        a_3 = 1.0 / (1.0 + np.exp(-z_3))
        h_x = [a_3[0], a_3[1], a_3[2]]
        h_x_output.append(h_x)
        p = np.argmax(h_x) + 1
        p_output.append(p)
    p_output = np.array(p_output)
    h_x_output = np.array(h_x_output)
    return p_output, h_x_output
