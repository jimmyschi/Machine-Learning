import numpy as np
from matplotlib import pyplot as plt

def computeCost(X,y,theta):
    #training sample m
    m = 4
    sum = 0
    for i in range(0,m):
        theta_T= np.transpose(theta)
        #print(theta_T)
        #print(X)
        h = np.matmul(theta_T,X)
        #print(h)
        dist = (h - y)
        #print(dist)
        ssd = dist * dist
        sum = sum + ssd
        #print(sum)
    #print(sum)
    cost = (sum/(2*m))
    #print("Cost: " + str(cost))
    return cost



#i)
#computeCost(np.array([[1],[1],[1]]),2,np.array([[0],[1],[.5]]))
#computeCost(np.array([[1],[2],[2]]),4,np.array([[0],[1],[.5]]))
#computeCost(np.array([[1],[3],[3]]),6,np.array([[0],[1],[.5]]))
#computeCost(np.array([[1],[4],[4]]),8,np.array([[0],[1],[.5]]))

#ii)
#computeCost(np.array([[1],[1],[1]]),2,np.array([[3.5],[0],[0]]))
#computeCost(np.array([[1],[2],[2]]),4,np.array([[3.5],[0],[0]]))
#computeCost(np.array([[1],[3],[3]]),6,np.array([[3.5],[0],[0]]))
#computeCost(np.array([[1],[4],[4]]),8,np.array([[3.5],[0],[0]]))

