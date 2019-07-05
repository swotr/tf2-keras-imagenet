import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import *

'''
- Used BN rather than the original channel normalization
- Normalization is applied before activation
- Merged two split branches into one
'''
def build_alexnet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = Conv2D(96, 11, 4, padding='same')(inputs)    
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, 2)(x)

    x = Conv2D(256, 5, 1, padding='same')(x)        
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, 2)(x)

    x = Conv2D(384, 3, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(384, 3, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(256, 3, 1, padding='same')(x)    
    x = BatchNormalization()(x)    
    x = ReLU()(x)
    x = MaxPooling2D(3, 2)(x)

    x = Flatten()(x)

    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1000)(x)
    outputs = Softmax()(x)

    return tf.keras.Model(inputs, outputs)