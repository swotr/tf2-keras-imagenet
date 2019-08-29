import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

def conv2d(filters, kernel_size, strides):
    return Conv2D(filters, kernel_size, strides=strides)

class MyModel(object):    
    def get_model(input_shape, training=False):
        inputs = Input(shape=input_shape)
        x = conv2d(32, 3, strides=4)(inputs)
        x = conv2d(32, 3, strides=4)(x)            
        x = Flatten()(x)
        x = Dense(128)(x)
        outputs = Dense(1000)(x)
        model = Model(inputs, outputs)
        return model