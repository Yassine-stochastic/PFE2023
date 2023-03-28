import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from time import time

# Set data type
DTYPE='float32'
#DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)
print('TensorFlow version used: {}'.format(tf.__version__))


# Final time
T = tf.constant(1., dtype=DTYPE)

# Spatial dimension
dim = 100

# Spatial domain of interest at t=0 (hyperrectangle)
a = np.zeros((dim), dtype=DTYPE)
b = np.ones((dim), dtype=DTYPE)

# Diffusion coefficient is assumed to be constant
sigma = np.sqrt(2, dtype=DTYPE)

# Define initial time condition
def fun_g(x):
    return tf.reduce_sum(tf.pow(x,2), axis=1, keepdims=True)

# Define exact reference solution
def fun_u(t, x):
    return tf.reduce_sum(tf.pow(x,2), axis=1, keepdims=True) + 2 * t * dim