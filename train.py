from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import time
import argparse
from tqdm import tqdm
from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
from tensorflow import keras

from imagenet_process import preprocess_for_train, preprocess_for_eval
#from efficientnet.model import *
from alexnet_keras import *
from model import EfficientNetB0

# Use NCHW

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_classes', default=1000, help='imagenet classes')
flags.DEFINE_integer('batch_size_per_replica', default=512, help='batch size per replica')
flags.DEFINE_integer('epochs', default=100, help='iterations, epochs, and batch_size are related')
#flags.DEFINE_string('data_format', default='channels_first', help=('data format NCHW instead of NHWC should be used'
#    'for a significant performance boost on GPU/TPU. NHWC should be used only if the network needs to be run on CPU'
#    'since the pooling operations are only supported on NHWC.'))
flags.DEFINE_integer('image_height', default=224, help='')
flags.DEFINE_integer('image_width', default=224, help='')
flags.DEFINE_float('initial_lr', default=0.016, help='initial_lr, 0.016 is default for batch_size_per_replica is 256 when using RMSProp')
flags.DEFINE_integer('num_threads', default=8, help='number of CPU threads for data processing')
flags.DEFINE_string('gpu', default='0', help='comma separated list of GPU(s)')
flags.DEFINE_string('test', default=None, help='run in test mode')

class ImageNetData(object):
    def __init__(self, data_dir):
        '''
        Read the list of tfrecord files for ImageNet classification.
        Serialized by https://github.com/tensorflow/models/tree/master/research/inception/inception/data
        '''
        self.train_files = list()
        self.val_files = list()
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.startswith('train-'):
                    self.train_files.append(os.path.join(root, file))
                elif file.startswith('validation-'):
                    self.val_files.append(os.path.join(root, file))
        print('# of train tfrecords {}'.format(len(self.train_files)))    
        print('# of val tfrecords {}'.format(len(self.val_files)))

    def get_train_files(self):
        return self.train_files

    def get_val_files(self):
        return self.val_files

class ImageNetModel(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def build_model(self):
        params = dict({
            'num_classes' : FLAGS.num_classes,
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
        })
        with self.strategy.scope():
            #model = EfficientNetB0(params).get_model((FLAGS.image_height, FLAGS.image_width, 3))
            model = build_alexnet((FLAGS.image_height, FLAGS.image_width, 3))
            model.summary()
            return model

class ImageNetTrainer(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def build_dataset(self, data, train_or_eval):
        assert train_or_eval in ['train', 'eval']        

        if train_or_eval == 'train':
            files = data.get_train_files()
            ds = tf.data.TFRecordDataset(files, buffer_size=64*1024*1024) # 64MB
            ds = ds.map(lambda x: preprocess_for_train(x, [FLAGS.image_height, FLAGS.image_width]),
                 num_parallel_calls=FLAGS.num_threads)
            #ds = ds.repeat()
            ds = ds.shuffle(10000)

        else:
            files = data.get_val_files()
            ds = tf.data.TFRecordDataset(files, buffer_size=64*1024*1024) # 64MB
            ds = ds.map(lambda x: preprocess_for_eval(x, [FLAGS.image_height, FLAGS.image_width]),
                 num_parallel_calls=FLAGS.num_threads)

        ds = ds.batch(FLAGS.batch_size_per_replica * self.strategy.num_replicas_in_sync)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def get_lr_scheduler(self, epoch):
        # numbers from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py        
        # decay factor = 0.97
        # decay epochs = 2.4
        # staircase = True
        # lr becomes half after 55 epochs
        # scaled_lr = FLAGS.initial_lr * ((FLAGS.batch_size_per_replica * self.strategy.num_replicas_in_sync) / 256)
        scaled_lr = FLAGS.initial_lr * ((FLAGS.batch_size_per_replica) / 256) # TODO
        lr = scaled_lr * (0.97**(epoch // 2.4)) # TODO
        return lr

    def get_loss(self):        
        '''
        For using multiple devices, losses 
         must be called within strategy scope.
        '''
        return keras.losses.SparseCategoricalCrossentropy()
    
    def get_optimizer(self):        
        '''
        For using multiple devices, optimizers 
         must be called within strategy scope.
        '''
        return keras.optimizers.RMSprop(FLAGS.initial_lr,
            rho=0.9, momentum=0.9, epsilon=0.001)

    def train(self, data, model, log_dir):
        with self.strategy.scope():
            model.compile(optimizer=self.get_optimizer(), 
                          loss=self.get_loss(),
                          metrics=[keras.metrics.SparseCategoricalAccuracy()])

        class DebugCallback(tf.keras.callbacks.Callback):            
            def __init__(self, strategy):
                super().__init__()
                self.strategy = strategy

            def on_train_batch_begin(self, batch, logs=None):
                if batch%10 == 0:
                    self.btime = time.time()

            def on_train_batch_end(self, batch, logs=None):
                if batch%10 == 0:
                    delta = time.time()-self.btime
                    images_per_sec = (FLAGS.batch_size_per_replica * self.strategy.num_replicas_in_sync) / delta
                    print('Training: batch {}, loss {:.4f} in {:.4f} images/sec'.format(
                        batch, logs['loss'], images_per_sec))

        callbacks = [
            keras.callbacks.TensorBoard(log_dir),
            keras.callbacks.LearningRateScheduler(self.get_lr_scheduler, verbose=1),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'model-{epoch:05d}.h5'),
                save_best_only=True,
                monitor='val_loss',
                verbose=1),
            DebugCallback(self.strategy),
        ]

        model.fit(self.build_dataset(data, 'train'),
            epochs=FLAGS.epochs,
            #steps_per_epoch=1281167//(FLAGS.batch_size_per_replica * self.strategy.num_replicas_in_sync),
            verbose=2, # 0=silent, 1=progress bar, 2=one line per epoch
            validation_data=self.build_dataset(data, 'eval'),
            validation_steps=50000//(FLAGS.batch_size_per_replica * self.strategy.num_replicas_in_sync), # TODO
            callbacks=callbacks)
                
class ImageNetTester(object):
    def __init__(self):
        pass

    def test(self, model_path, test_dir):
        pass

def main(argv):
    if FLAGS.gpu:
        #os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        # The best practice for using multiple GPUs (TF2)
        # with tf.distribute.Strategy():
        #     model = keras.Model(inputs, outputs)
        #     keras.compile(optimizer, loss, metrics)
        # model.fit()
        devices = list()
        for id in list(map(int, FLAGS.gpu.split(','))):
            devices.append('/gpu:{}'.format(id))
        print(devices)
        strategy = tf.distribute.MirroredStrategy(devices=devices)        

    if FLAGS.test:
        pass
    else:
        data = ImageNetData('/media/jihyeony/hdd1/minje/ImageNet/tfrecord/')
        model = ImageNetModel(strategy).build_model()
        fname = os.path.basename(sys.argv[0]) # argv[0] contains python filename
        fname = fname[:fname.rfind('.')]
        log_dir = os.path.join('train_log/', fname)
        if not os.path.exists(log_dir):
            print(log_dir)
            os.makedirs(log_dir)
        trainer = ImageNetTrainer(strategy)
        trainer.train(data, model, log_dir)

if __name__ == '__main__':
    app.run(main)