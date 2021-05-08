from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D

import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import scipy.stats as st

def _get_kernel(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def gaussian_kernel():
  gaussian_kernel = np.float32(_get_kernel(31, 4))
  gaussian_kernel = gaussian_kernel[...,np.newaxis, np.newaxis]
  return np.asarray([gaussian_kernel] )


# Not working
def min_max_norm(x):
    shape = x.shape


    x1 = tf.expand_dims(tf.expand_dims(K.max(K.max(x, axis=3), axis=2), 2), 3)
    print("problem in broadcast")
    print(x1.shape, 'AND', shape)
    x1 = tf.broadcast_to(x1, shape)
    
    print('problem in broadcast solved')
    x2 =  tf.broadcast_to(tf.expand_dims(tf.expand_dims(K.min(K.min(x, axis=3), axis=2), 2), 3), shape)
    
    return tf.math.divide(x-x2, x1-x2+1e-8)
    

def SA(attention, x):

    tmp_input = Input(attention.shape[1:])
    pad1 = ZeroPadding2D(padding=15)(tmp_input)
    conv2 = Conv2D(filters=1, kernel_size=31, use_bias=False)(pad1)
    conv = Model(tmp_input, conv2)

    conv.layers[-1].set_weights(gaussian_kernel())

    soft_attention = conv(attention)
    return soft_attention
    
    
    # To be added in the future 
    soft_attention = min_max_norm(soft_attention)       # normalization
    m = tf.math.maximum(soft_attention, attention)
    ret = tf.math.multiply(x, m)    # mul
    return ret