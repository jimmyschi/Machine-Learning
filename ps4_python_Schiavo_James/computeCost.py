import numpy as np
from matplotlib import pyplot as plt

def computeCost(X,y,theta):
    #training sample m
    m = X.shape[0]
    sum = 0
    h = np.matmul(X,theta)
    dist = (h - y)**2
    sum = sum + np.sum(dist)
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

