import numpy as np
import scipy.io as sio
from  sklearn.linear_model  import  LogisticRegression

def logReg_multi(X_train,y_train,X_test):
    y_predict = np.zeros((X_test.shape[0],1))
    print(y_predict.shape)
    y_train1 = np.zeros((y_train.shape[0],1))
    y_train2 = np.zeros((y_train.shape[0],1))
    y_train3 = np.zeros((y_train.shape[0],1))
    print("y_train: " + str(y_train))
    for i in range(y_train.shape[0]):
        if y_train[i] == 1:
            y_train1[i] = 1
        elif y_train[i] == 2:
            y_train2[i] = 1
        elif y_train[i] == 3:
            y_train3[i] = 1
    print("y_train1: " + str(y_train1))
    print("y_train2: " + str(y_train2))
    print("y_train3: " + str(y_train3))
    mdl1 = LogisticRegression(random_state=0).fit(X_train,y_train1)
    y_pred1 = mdl1.predict_proba(X_test)
    mdl2 = LogisticRegression(random_state=0).fit(X_train,y_train2)
    y_pred2 = mdl2.predict_proba(X_test)
    mdl3 = LogisticRegression(random_state=0).fit(X_train,y_train3)
    y_pred3 = mdl3.predict_proba(X_test)
    for i in range(y_predict.shape[0]):
        index = np.array([y_pred1[i][1],y_pred2[i][1],y_pred3[i][1]])
        y_predict[i] = np.argmax(index) + 1
    print("y_predict: " + str(y_predict))
    return y_predict


