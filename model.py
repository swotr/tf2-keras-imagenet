from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
import tensorflow.keras.models as KM
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects

'''
Initializers
'''

class EN_Conv2DKernelInitializer(Initializer):
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """

    def __call__(self, shape, dtype=K.floatx(), **kwargs):
        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random.normal(shape, mean=0.0, 
            stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

class EN_DenseKernelInitializer(Initializer):
    """
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    Args:
      shape: shape of variable
      dtype: dtype of variable

    Returns:
      an initialization for the variable
    """

    def __call__(self, shape, dtype=K.floatx(), **kwargs):        
        init_range = 1.0 / np.sqrt(shape[1])
        return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)

get_custom_objects().update({
    'EN_Conv2DKernelInitializer' : EN_Conv2DKernelInitializer,
    'EN_DenseKernelInitializer' : EN_DenseKernelInitializer,
})

'''
Custom layers
'''

class Swish(Layer):
    def call(self, inputs):
        #return tf.nn.swish(inputs)
        return (tf.math.sigmoid(inputs) * inputs)

class DropConnect(Layer):
    def __init__(self, drop_connect_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training):
        def _drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.math.divide(inputs, keep_prob) * binary_tensor
            return output

        return K.in_train_phase(_drop_connect, inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config['drop_connect_rate'] = self.drop_connect_rate
        return config

get_custom_objects().update({
    'DropConnect' : DropConnect,
    'Swish' : Swish,
})

'''
Model (Keras Model functional API)
'''

class EfficientNetB0(object):
    def __init__(self, params):
        self.params = params

    def conv2d(self, filters, kernel_size, strides):
        return Conv2D(filters, kernel_size, strides=strides,
                kernel_initializer=EN_Conv2DKernelInitializer(),
                padding='same', use_bias=False)

    def dwconv2d(self, kernel_size, strides):
        return DepthwiseConv2D(kernel_size, strides=strides,
                depthwise_initializer=EN_Conv2DKernelInitializer(),
                padding='same', use_bias=False)

    def bn(self):
        return BatchNormalization(axis=-1,
                momentum=self.params['batch_norm_momentum'],
                epsilon=self.params['batch_norm_epsilon'])

    def SEBlock(self, filters):
        def _block(inputs):
            x = inputs
            x = Lambda(lambda y : tf.keras.backend.mean(y, axis=[1, 2], keepdims=True))(x)
            x = self.conv2d(filters, 1, 1)(x)
            #x = Swish()(x)
            x = ReLU()(x)

            x = self.conv2d(filters, 1, 1)(x)
            x = Activation('sigmoid')(x)

            x = Multiply()([x, inputs])
            return x

        return _block

    def MBConvBlock(self, filters, kernel_size, strides, output_filters, id_skip):
        def _block(inputs):
            x = inputs
            x = self.conv2d(filters, 1, 1)(x)
            x = self.bn()(x)
            #x = Swish()(x)
            x = ReLU()(x)

            x = self.dwconv2d(kernel_size, strides)(x)
            x = self.bn()(x)
            #x = Swish()(x)
            x = ReLU()(x)

            x = self.SEBlock(filters)(x)

            # output phase
            x = self.conv2d(output_filters, 1, 1)(x)
            x = self.bn()(x)

            if id_skip:
                # only apply DropConnect if skip presents
                if self.params['drop_connect_rate']:
                    x = DropConnect(self.params['drop_connect_rate'])(x)
                x = Add()([x, inputs])            

            return x

        return _block            

    def get_model(self, input_shape):
        # stem part
        inputs = Input(input_shape)
        x = inputs
        x = self.conv2d(32, 3, 2)(x)
        x = self.bn()(x)
        #x = Swish()(x)
        x = ReLU()(x)        

        # blocks part        
        for i in range(len(self.params['MB_blocks']['repeats'])):
            repeat = self.params['MB_blocks']['repeats'][i]
            filters = self.params['MB_blocks']['filters'][i]
            kernel_size = self.params['MB_blocks']['kernel_sizes'][i]
            strides = self.params['MB_blocks']['strides'][i]
            output_filters = self.params['MB_blocks']['output_filters'][i]
            id_skip = self.params['MB_blocks']['id_skip'][i]
            for _ in range(repeat):
                x = self.MBConvBlock(filters, kernel_size, 
                    strides, output_filters, id_skip)(x)

        # head part
        x = self.conv2d(1280, 1, 1)(x)
        x = self.bn()(x)
        #x = Swish()(x)
        x = ReLU()(x)
        
        # top part
        x = GlobalAveragePooling2D()(x)
        if self.params['dropout_rate'] > 0:
            x = Dropout(self.params['dropout_rate'])(x)
        x = Dense(self.params['num_classes'], 
                kernel_initializer=EN_DenseKernelInitializer())(x)        
        outputs = Activation('softmax')(x)

        return KM.Model(inputs, outputs)

def main(argv):    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    channel_first = True
    input_shape = [224, 224, 3] if channel_first is False else [3, 224, 224]
    model = EfficientNetB0(params = dict({
            'num_classes' : 1000,
            'MB_blocks' : {
                    'repeats' : [1, 2, 2, 3, 3, 4, 1],
                    'filters' : [16, 24, 40, 80, 112, 192, 320],
                    'kernel_sizes' : [3, 3, 5, 3, 5, 5, 3],
                    'strides' : [1, 2, 2, 1, 2, 2, 1],
                    'output_filters' : [16, 24, 40, 80, 112, 192, 320],                    
                    'id_skip' : [False, False, False, False, False, False, False],
                },
            'batch_norm_momentum' : 0.99,
            'batch_norm_epsilon' : 2e-5,
            'drop_connect_rate' : 0.2,            
            'dropout_rate' : 0.2,
        })).get_model(input_shape)
    model.summary()
    #tf.keras.experimental.export_saved_model(model, flags.FLAGS.save_dir)
    import onnx
    import keras2onnx
    #import onnxmltools
    #onnx_model = onnxmltools.convert_keras(model, target_opset=7)
    onnx_model = keras2onnx.convert_keras(model)
    onnx.save_model(onnx_model, 'onnx/model.onnx')

if __name__ == '__main__':
    app.run(main)