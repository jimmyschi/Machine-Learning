import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import cv2 as cv 
import random
from torch.utils.data import random_split
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from kmeans_single import kmeans_single
from kmeans_multiple import kmeans_multiple

#1a
data = sio.loadmat("./input/HW8_data1.mat")
X = np.array(data["X"])
y = np.array(data["y"])
idx = np.linspace(0,4999,5000)
rand_img = np.random.choice(idx, 25)
print("rand_img: " + str(rand_img))
img =  []
for i in range(len(rand_img)):
    img.append(np.resize(X[int(rand_img[i]),:], (20, 20)))
    #print(X[int(rand_img[i]),:].shape)
img = np.array(img)
img = np.resize(img, (5, 5, 20, 20))
img = np.transpose(img, axes=[0, 2, 1, 3])
img = np.resize(img,(100,100))
cv.imwrite("./output/ps8-1-a-1.png",img*255)

#1b
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.10)

#1c
X_1,X_2,X_3,X_4,X_5 = random_split(X_train,[900,900,900,900,900])
X_1 = np.array(X_1)
X_2 = np.array(X_2)
X_3 = np.array(X_3)
X_4 = np.array(X_4)
X_5 = np.array(X_5)
y_1 = np.zeros((900,1)) 
y_2 = np.zeros((900,1)) 
y_3 = np.zeros((900,1)) 
y_4 = np.zeros((900,1)) 
y_5 = np.zeros((900,1)) 
for i in range(X_train.shape[0]):
    for j in range(900):
        if np.array_equal(X_train[i, :], X_1[j, :]):
            y_1[j] = y_train[i]
            break    
        elif np.array_equal(X_train[i, :], X_2[j, :]):
            y_2[j] = y_train[i]
            break
        elif np.array_equal(X_train[i, :], X_3[j, :]):
            y_3[j] = y_train[i]
            break
        elif np.array_equal(X_train[i, :], X_4[j, :]):
            y_4[j] = y_train[i]
            break
        elif np.array_equal(X_train[i, :], X_5[j, :]):
            y_5[j] = y_train[i]
            break
y_1 = np.array(y_1)
y_2 = np.array(y_2)
y_3 = np.array(y_3)
y_4 = np.array(y_4)
y_5 = np.array(y_5)
#print("y_1: " + str(y_1))
np.save("./input/X_1",X_1)
np.save("./input/X_2",X_2)
np.save("./input/X_3",X_3)
np.save("./input/X_4",X_4)
np.save("./input/X_5",X_5)



#1d
print("SVM CLASSIFIER!!!!")
svm_classifier = svm.SVC(kernel='rbf',decision_function_shape='ovo')
svm_classifier.fit(X_1,y_1)
y_pred1 = svm_classifier.predict(X_test)
accuracy1 = metrics.accuracy_score(y_test,y_pred1)
print("testing accuracy: " + str(accuracy1))
y_pred = svm_classifier.predict(X_1)
accuracy1 = metrics.accuracy_score(y_1,y_pred)
print("training X_1 accuracy: " + str(accuracy1))
y_pred = svm_classifier.predict(X_2)
accuracy1 = metrics.accuracy_score(y_2,y_pred)
print("training X_2 accuracy: " + str(accuracy1))
y_pred = svm_classifier.predict(X_3)
accuracy1 = metrics.accuracy_score(y_3,y_pred)
print("training X_3 accuracy: " + str(accuracy1))
y_pred = svm_classifier.predict(X_4)
accuracy1 = metrics.accuracy_score(y_4,y_pred)
print("training X_4 accuracy: " + str(accuracy1))
y_pred = svm_classifier.predict(X_5)
accuracy1 = metrics.accuracy_score(y_5,y_pred)
print("training X_5 accuracy: " + str(accuracy1))

#1e
print("KNN!!!!")
knn = KNeighborsClassifier(5)
knn.fit(X_2,y_2)
y_pred2 = knn.predict(X_test)
accuracy1 = metrics.accuracy_score(y_test,y_pred2)
print("testing accuracy: " + str(accuracy1))
y_pred = knn.predict(X_1)
accuracy1 = metrics.accuracy_score(y_1,y_pred)
print("training X_1 accuracy: " + str(accuracy1))
y_pred = knn.predict(X_2)
accuracy1 = metrics.accuracy_score(y_2,y_pred)
print("training X_2 accuracy: " + str(accuracy1))
y_pred = knn.predict(X_3)
accuracy1 = metrics.accuracy_score(y_3,y_pred)
print("training X_3 accuracy: " + str(accuracy1))
y_pred = knn.predict(X_4)
accuracy1 = metrics.accuracy_score(y_4,y_pred)
print("training X_4 accuracy: " + str(accuracy1))
y_pred = knn.predict(X_5)
accuracy1 = metrics.accuracy_score(y_5,y_pred)
print("training X_5 accuracy: " + str(accuracy1))

#1f
print("LOGISTIC REGRESSION!!!!")
log_reg = LogisticRegression()
log_reg.fit(X_3,y_3)
y_pred3 = log_reg.predict(X_test)
accuracy1 = metrics.accuracy_score(y_test,y_pred3)
print("testing accuracy: " + str(accuracy1))
y_pred = log_reg.predict(X_1)
accuracy1 = metrics.accuracy_score(y_1,y_pred)
print("training X_1 accuracy: " + str(accuracy1))
y_pred = log_reg.predict(X_2)
accuracy1 = metrics.accuracy_score(y_2,y_pred)
print("training X_2 accuracy: " + str(accuracy1))
y_pred = log_reg.predict(X_3)
accuracy1 = metrics.accuracy_score(y_3,y_pred)
print("training X_3 accuracy: " + str(accuracy1))
y_pred = log_reg.predict(X_4)
accuracy1 = metrics.accuracy_score(y_4,y_pred)
print("training X_4 accuracy: " + str(accuracy1))
y_pred = log_reg.predict(X_5)
accuracy1 = metrics.accuracy_score(y_5,y_pred)
print("training X_5 accuracy: " + str(accuracy1))

#1g
print("DECISION TREE!!!!")
tree = DecisionTreeClassifier()
tree.fit(X_4,y_4)
y_pred4 = tree.predict(X_test)
accuracy1 = metrics.accuracy_score(y_test,y_pred4)
print("testing accuracy: " + str(accuracy1))
y_pred = tree.predict(X_1)
accuracy1 = metrics.accuracy_score(y_1,y_pred)
print("training X_1 accuracy: " + str(accuracy1))
y_pred = tree.predict(X_2)
accuracy1 = metrics.accuracy_score(y_2,y_pred)
print("training X_2 accuracy: " + str(accuracy1))
y_pred = tree.predict(X_3)
accuracy1 = metrics.accuracy_score(y_3,y_pred)
print("training X_3 accuracy: " + str(accuracy1))
y_pred = tree.predict(X_4)
accuracy1 = metrics.accuracy_score(y_4,y_pred)
print("training X_4 accuracy: " + str(accuracy1))
y_pred = tree.predict(X_5)
accuracy1 = metrics.accuracy_score(y_5,y_pred)
print("training X_5 accuracy: " + str(accuracy1))

#1h
print("RANDOM FOREST!!!!")
forest = RandomForestClassifier(50)
forest.fit(X_5,y_5)
y_pred5 = forest.predict(X_test)
accuracy1 = metrics.accuracy_score(y_test,y_pred5)
print("testing accuracy: " + str(accuracy1))
y_pred = forest.predict(X_1)
accuracy1 = metrics.accuracy_score(y_1,y_pred)
print("training X_1 accuracy: " + str(accuracy1))
y_pred = forest.predict(X_2)
accuracy1 = metrics.accuracy_score(y_2,y_pred)
print("training X_2 accuracy: " + str(accuracy1))
y_pred = forest.predict(X_3)
accuracy1 = metrics.accuracy_score(y_3,y_pred)
print("training X_3 accuracy: " + str(accuracy1))
y_pred = forest.predict(X_4)
accuracy1 = metrics.accuracy_score(y_4,y_pred)
print("training X_4 accuracy: " + str(accuracy1))
y_pred = forest.predict(X_5)
accuracy1 = metrics.accuracy_score(y_5,y_pred)
print("training X_5 accuracy: " + str(accuracy1))

#i    
y_pred = np.zeros(y_test.shape)
print("y_test shape: " + str(y_test.shape))
print("y_pred shape: " + str(y_pred.shape))
for i in range(y_pred.shape[0]):
    predictions = [y_pred1[i],y_pred2[i],y_pred3[i],y_pred4[i],y_pred5[i]]
    print("predictions: " + str(predictions))
    y_pred[i] = max(predictions,key=predictions.count)
    print("p_pred[i]: "  + str(y_pred[i]))
accuracy  = metrics.accuracy_score(y_test,y_pred)
print("majority vote accuracy: " + str(accuracy))

#2
def Segment_kmeans(im_in,K,iters,R):
    im_in = cv.normalize(im_in.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    im_in = cv.resize(im_in,(100,100))
    X = np.reshape(im_in,(im_in.shape[0]*im_in.shape[1],3))
    ids,means,ssd = kmeans_multiple(X,K,iters,R)
    print("ids: " + str(ids))
    print("means: " + str(means))
    print("ssd: " + str(means))
    print(means)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(K):
                k_val = k + 1
                if ids[i] == k_val:
                    X[i][j] = means[k][j]
    im_out = cv.convertScaleAbs(X, alpha=(255.0/X.max()))
    im_out = np.reshape(im_out,im_in.shape)
    print("im_out: " + str(im_out))
    return im_out



image1 = cv.imread("./input/HW8_F22_images/im1.jpg")
image2 = cv.imread("./input/HW8_F22_images/im2.jpg")
image3 = cv.imread("./input/HW8_F22_images/im3.png")


K = [3,5,7]
iters = [7,15,30]
R = [5,15,20]
count = 1

for k in K:
    for it in iters:
        for r in R:
            image1_out = Segment_kmeans(image1,k,it,r)
            cv.imwrite("./output/ps8-2-c-" + str(count) + ".jpg",image1_out)
            count = count + 1
            image2_out = Segment_kmeans(image2,k,it,r)
            cv.imwrite("./output/ps8-2-c-" + str(count) + ".jpg",image2_out)
            count = count + 1
            image3_out = Segment_kmeans(image3,k,it,r)
            cv.imwrite("./output/ps8-2-c-" + str(count) + ".png",image3_out)
            count = count + 1


