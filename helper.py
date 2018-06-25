import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


# LEAKY RELU UNIT
def lrelu(x):
    return tf.maximum(0.2*x,x)


# GENERATE DILATED LAYER FROM 1D SIGNAL
def signal_to_dilated(signal, dilation, n_channels):
    shape = tf.shape(signal)
    pad_elements = dilation - 1 - (shape[2] + dilation - 1) % dilation
    dilated = tf.pad(signal, [[0, 0], [0, 0], [0, pad_elements], [0, 0]])
    dilated = tf.reshape(dilated, [shape[0],-1,dilation,n_channels])
    return tf.transpose(dilated, perm=[0,2,1,3]), pad_elements


# COLLAPSE DILATED LAYER TO 1D SIGNAL
def dilated_to_signal(dilated, pad_elements, n_channels):
    shape = tf.shape(dilated)
    signal = tf.transpose(dilated, perm=[0,2,1,3])
    signal = tf.reshape(signal, [shape[0],1,-1,n_channels])
    return signal[:,:,:shape[1]*shape[2]-pad_elements,:]


# ADAPTIVE BATCH NORMALIZATION LAYER
def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)


# IDENTITY INITIALIZATION OF CONV LAYERS
def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer


# L1 LOSS FUNCTION
def l1_loss(target,current):
    return tf.reduce_mean(tf.abs(target-current))


# L2 LOSS FUNCTION
def l2_loss(target,current):
    return tf.reduce_mean(tf.square(target-current))


