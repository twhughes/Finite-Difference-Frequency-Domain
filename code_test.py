from NN import NN
import numpy as np
import matplotlib.pylab as plt

layer_sizes = [2,4,1]
activations = ['relu','sigmoid']
N = NN(layer_sizes,activations)

#print(N.biases[4].shape)
input = np.array([[1,2,3],[3,5,4]])

N.forward_prop(input)
N.back_prop(np.array([[1,2,3]]))
N.derivative_check(m=6,verbose=False)
N.update_weights()


