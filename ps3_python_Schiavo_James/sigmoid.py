import numpy as np
import math
def sigmoid(z):
    print("z shape: " + str(z.shape))
    g = np.ones((z.shape[0],1))
    g = 1/(1 + np.exp(-z))
    #print(g)
    return g