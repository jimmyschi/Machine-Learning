import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gradientDescent import gradientDescent

data_size = open("./input/hw2_data2.txt","r+")
size = len(data_size.readlines())
features = np.zeros((size,3))
data_size.close()
data = open("./input/hw2_data2.txt","r+")
count = 0
for lines in data:
    print(lines)
    line_split = lines.split(",")
    features[count][0] = int(line_split[0])
    features[count][1] = int(line_split[1])
    features[count][2] = int(line_split[2])
    count = count + 1
print(features)
data.close()
size_house = features[:,0]
#print(size_house.shape)
bedrooms = features[:,1]
#print(bedrooms.shape)
price = features[:,2]
#print(price.shape)

#standardize data
mean_size = np.mean(size_house)
print("mean_size:" + str(mean_size))
mean_bedrooms = np.mean(bedrooms)
print("mean bedrooms: " + str(mean_bedrooms))
std_size = np.std(size_house)
print("std size: " + str(std_size))
std_bedrooms = np.std(bedrooms)
print("std bedrooms: " + str(std_bedrooms))
for i in range(features.shape[0]):
    size_house[i] = (size_house[i] - mean_size)/std_size
    bedrooms[i] = (bedrooms[i] - mean_bedrooms)/std_bedrooms

#feature matrix X
X = np.ones((size_house.shape[0],3))
for i in range(X.shape[0]):
    X[i][1] = size_house[i]
    X[i][2] = bedrooms[i]
print("X: " + str(X))
print(X.shape)
y = price
print(y.shape)

#5b
theta,cost = gradientDescent(X,y,.01,750)
print(theta)
plt.plot(cost)
plt.savefig("./output/ps2-5-b.png")

#5c
norm_size_pred = (1250 - mean_size)/std_size
norm_bedroom_pred = (3 - mean_bedrooms)/std_bedrooms
pred_price = theta[0] + theta[1]*norm_size_pred + theta[2]*norm_bedroom_pred
print("Predicted price: " + str(pred_price))