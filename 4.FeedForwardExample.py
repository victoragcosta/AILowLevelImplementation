import matplotlib.pylab as plt
import numpy as np
import time

def f(x):
    return 1 / (1 + np.exp(-x))

def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        # Choose the input for each layer
        # x for layer 0, the previous one for the other ones
        if l == 0:
            node_in = x
        else:
            node_in = h
        # Setup the output for the next layer
        h = np.zeros((w[l].shape[0],))
        # Loop through the rows (each represents a node in l+1)
        for i in range(w[l].shape[0]):
            # Setup sum for f input
            f_sum = 0
            # Loop through the columns (each represents a node in l)
            for j in range(w[l].shape[1]):
                f_sum += w[l][i][j] * node_in[j] # input from node j from l times w_ij
            # Add the bias
            f_sum += b[l][i]
            # Apply the activation function to the sum
            h[i] = f(f_sum)
    return h

def matrix_feed_forward_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = f(z)
    return h

w1 = np.array([[0.2,0.2,0.2], [0.4,0.4,0.4], [0.6,0.6,0.6]])
w2 = np.zeros((1,3))
w2[0,:] = np.array([0.5,0.5,0.5])
b1 = np.array([0.8,0.8,0.8])
b2 = np.array([0.2])
w = [w1, w2]
b = [b1, b2]
x = [1.5, 2.0, 3.0]

before = time.time()
looped = simple_looped_nn_calc(3,x,w,b)
middle = time.time()
matrix = matrix_feed_forward_calc(3, x, w, b)
after = time.time()
time_looped = middle-before
print('looped got', looped, 'in', time_looped)
time_matrix = after-middle
print('matrix got', matrix, 'in', time_matrix)