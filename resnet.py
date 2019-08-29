import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

def resnet_layer(filters, kernel_size, strides,
                 activation='relu', batch_normalization=True, conv_first=True):
    def _wrapper(inputs):
        conv = Conv2D(filters, kernel_size, strides=strides, padding='same', 
            kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x
    return _wrapper

def build_resnet(input_shape, n=2, num_classes=10):
    # https://keras.io/examples/cifar10_resnet/
    # V2 only
    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------
    depth = n * 9 + 2    
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2')
    filters = 16
    num_res_blocks = int((depth - 2) / 9)
    
    inputs = Input(shape=input_shape)

    # Stem part    
    x = resnet_layer(filters, 3, 1, 'relu', 
        batch_normalization=True, conv_first=True)(inputs)

    # Main blocks
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                out_filters = filters * 4
                if res_block == 0: # first layer and first stage
                    activation = None
                    batch_normalization = False                    
            else:
                out_filters = filters * 2
                if res_block == 0:
                    strides = 2 # downscale
            
            y = resnet_layer(filters, 1, strides, activation, 
                batch_normalization, conv_first=False)(x)
            y = resnet_layer(filters, 3, 1, 'relu', 
                batch_normalization=True, conv_first=False)(y)
            y = resnet_layer(out_filters, 1, 1, 'relu', 
                batch_normalization=True, conv_first=False)(y)
            if res_block == 0:
                x = resnet_layer(out_filters, 1, strides, activation=None,
                    batch_normalization=False, conv_first=True)(x)
            x = tf.keras.layers.add([x, y])

        filters = out_filters

    # Head part
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, 
                    activation='softmax', 
                    kernel_initializer='he_normal')(y)

    model = tf.keras.Model(inputs, outputs)
    return model

def build_resnet20_nas(input_shape, config, num_classes=10):
    '''
    Assume
    n = 2
    config:
        filters = 8, 16, 24
        kenel = 3, 5
        filter_multiplier = 4, 2
    '''
    n = 2    
    depth = n * 9 + 2    
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2')    
    filters = config['base_filters'] # NAS parameter
    ks = config['kernel'] # NAS parameter
    fm = config['filter_multiplier'] # NAS parameter
    num_res_blocks = int((depth - 2) / 9)
    
    inputs = Input(shape=input_shape)

    # Stem part    
    x = resnet_layer(filters, 3, 1, 'relu', 
        batch_normalization=True, conv_first=True)(inputs)

    # Main blocks
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                out_filters = filters * fm
                if res_block == 0: # first layer and first stage
                    activation = None
                    batch_normalization = False                    
            else:
                out_filters = filters * 2
                if res_block == 0:
                    strides = 2 # downscale
            
            y = resnet_layer(filters, 1, strides, activation, 
                batch_normalization, conv_first=False)(x)
            y = resnet_layer(filters, ks, 1, 'relu', 
                batch_normalization=True, conv_first=False)(y)
            y = resnet_layer(out_filters, 1, 1, 'relu', 
                batch_normalization=True, conv_first=False)(y)
            if res_block == 0:
                x = resnet_layer(out_filters, 1, strides, activation=None,
                    batch_normalization=False, conv_first=True)(x)
            x = tf.keras.layers.add([x, y])

        filters = out_filters

    # Head part
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, 
                    activation='softmax', 
                    kernel_initializer='he_normal')(y)

    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = build_resnet(input_shape=[224, 224, 3], n=2, num_classes=1000)
    model.summary()