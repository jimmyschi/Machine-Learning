import numpy as np 

def sigmoidGradient(z):
    z = np.array(z)
    g_z = 1.0 / (1.0 + np.exp(-z))
    g_prime = g_z*(1-g_z)
    return g_prime