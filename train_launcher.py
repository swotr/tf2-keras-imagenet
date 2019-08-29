import os
import sys
import time

from tqdm import tqdm
from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import tensorflow.keras

from model import EfficientNetB0
from resnet import build_resnet20_nas
from train_cifar import Cifar10, Cifar100, CifarTrainer

FLAGS = flags.FLAGS
flags.DEFINE_string('log_group', default='resnet20_nas', help='log group name')
flags.DEFINE_boolean('full_data', default=True, help='use full training data')

class launcher(object):
    def __init__(self, strategy):
        self.strategy = strategy
        valid_train_ids = None
        label_map = None
        if FLAGS.full_data is not True:        
            valid_train_ids = np.load('/home/minje/dev/jnote/valid_train_ids.npy')
            label_map = np.load('/home/minje/dev/jnote/label_map.npy')            
        if FLAGS.num_classes == 10:
            self.data = Cifar10('/home/minje/dev/dataset/cifar/cifar-10-batches-py/',
                valid_ids=valid_train_ids)
        elif FLAGS.num_classes == 100:
            self.data = Cifar100('/home/minje/dev/dataset/cifar/cifar100/cifar-100-python/',
                valid_ids=valid_train_ids,
                label_map=label_map)

    def __call__(self, config):
        # build model from config
        with self.strategy.scope():
            input_shape = [32, 32, 3]
            self.model = build_resnet20_nas(input_shape, config, num_classes=FLAGS.num_classes)
            # self.model = EfficientNetB0(config).get_model(input_shape, no_strides_at_first=True)
            self.model.summary()

        # prepare trainer
        logdir = os.path.join('train_log/', FLAGS.log_group)
        stamp = time.strftime('%Y%B%d-%H%M%S', time.gmtime())
        logdir = os.path.join(logdir, stamp)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.trainer = CifarTrainer(self.strategy, self.data, self.model, 
            logdir=logdir, compiled=False)        
        self.trainer.train()

def main(argv):
    # prepare training resource
    if FLAGS.gpu is not None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
                exit(0)

        devices = list()
        for id in list(map(int, FLAGS.gpu.split(','))):
            devices.append('/gpu:{}'.format(id))
        print(devices)
        strategy = tf.distribute.MirroredStrategy(devices=devices)

        lx = launcher(strategy=strategy)

        # prepare network configurations (ResNet-20)
        configs = [
            dict({'base_filters': 8, 'kernel': 3, 'filter_multiplier': 2}),
            dict({'base_filters': 8, 'kernel': 3, 'filter_multiplier': 4}),
            dict({'base_filters': 8, 'kernel': 5, 'filter_multiplier': 2}),
            dict({'base_filters': 8, 'kernel': 5, 'filter_multiplier': 4}),
            dict({'base_filters': 16, 'kernel': 3, 'filter_multiplier': 2}),
            dict({'base_filters': 16, 'kernel': 3, 'filter_multiplier': 4}),
            dict({'base_filters': 16, 'kernel': 5, 'filter_multiplier': 2}),
            dict({'base_filters': 16, 'kernel': 5, 'filter_multiplier': 4}),
            dict({'base_filters': 24, 'kernel': 3, 'filter_multiplier': 2}),
            dict({'base_filters': 24, 'kernel': 3, 'filter_multiplier': 4}),
            dict({'base_filters': 24, 'kernel': 5, 'filter_multiplier': 2}),
            dict({'base_filters': 24, 'kernel': 5, 'filter_multiplier': 4}),
        ]
        for cfg in configs:
            lx(cfg)
        # prepare network configurations (EFN0)
        # base_config = dict({
        #     'num_classes' : FLAGS.num_classes,
        #     'MB_blocks' : {
        #             'repeats' : [1, 2, 2, 3, 3, 4, 1],
        #             'in_channels' : [32, 16, 24, 40, 80, 112, 192],
        #             'expand_ratio' : [1, 6, 6, 6, 6, 6, 6],
        #             'kernel_sizes' : [3, 3, 5, 3, 5, 5, 3],
        #             'strides' : [1, 2, 2, 2, 1, 2, 1], # differnt from paper (comply with official implementation)
        #             'output_filters' : [16, 24, 40, 80, 112, 192, 320],
        #             'id_skip' : [True, True, True, True, True, True, True],
        #         },
        #     'drop_connect_rate' : 0.2,            
        #     'dropout_rate' : 0.2})
        # repeats_configs = [
        #     dict({'repeats' : [1, 2, 2, 2, 2, 3, 1]}),
        #     dict({'repeats' : [1, 2, 2, 3, 3, 4, 1]})
        # ]
        # kernels_configs = [
        #     dict({'kernel_sizes' : [3, 3, 3, 3, 3, 3, 3]}),
        #     dict({'kernel_sizes' : [3, 3, 5, 3, 5, 5, 3]}),
        # ]
        # channels_configs = [
        #     dict({'in_channels' : [18, 16, 16, 24, 48, 68, 116],
        #           'output_filters' : [16, 16, 24, 48, 68, 116, 192]}),
        #     dict({'in_channels' : [24, 16, 20, 32, 64, 90, 152],
        #           'output_filters' : [16, 20, 32, 64, 90, 152, 256]}),
        #     dict({'in_channels' : [32, 16, 24, 40, 80, 112, 192],
        #           'output_filters' : [16, 24, 40, 80, 112, 192, 320]})
        # ] 
        # for rx in repeats_configs:
        #     for kx in kernels_configs:
        #         for cx in channels_configs:
        #             config = dict(base_config)
        #             config['MB_blocks'].update(rx)
        #             config['MB_blocks'].update(kx)
        #             config['MB_blocks'].update(cx)
        #             lx(config)


if __name__ == '__main__':
    app.run(main)

