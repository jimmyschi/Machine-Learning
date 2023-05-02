from re import X
import numpy as np

def normalEqn(X_train,y_train):
    theta = np.zeros((X_train.shape[1],1))
    #print(X_train)
    X_t = np.transpose(X_train)
    #print(X_t)
    X_t_X = np.matmul(X_t,X_train)
    #print(X_t_X)
    X_t_X_inv = np.linalg.pinv(X_t_X)
    X = np.matmul(X_t_X_inv,X_t)
    ans = np.matmul(X,y_train)
    print(ans)
    return theta


#X_train = np.array([[1,1,1],[1,2,2],[1,3,3],[1,4,4]])
#y_train = np.array([[2],[4],[6],[8]])
#theta = normalEqn(X_train,y_train)
#print("theta: " + str(theta))