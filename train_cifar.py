import os, sys
import time
import argparse
from tqdm import tqdm
from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
from tensorflow import keras

from skimage import io
from skimage.util import img_as_float32

from model import EfficientNetB0
from alexnet_keras import *
from resnet import *

FLAGS = flags.FLAGS
# parameters
flags.DEFINE_boolean('channels_first', default=False, help='channels_first (NCHW) for faster GPU compute')
flags.DEFINE_integer('num_classes', default=10, help='number of output classes')
flags.DEFINE_float('weight_decay', default=2e-4, help='Weight decay for l2 regularization')
flags.DEFINE_float('label_smoothing', default=0.1, help='Label smoothing parameter for softmax_cross_entropy') # TODO(not implemented)
flags.DEFINE_integer('batch_size_per_replica', default=32, help='batch size per replica')
flags.DEFINE_integer('epochs', default=100, help='iterations, epochs, and batch_size are related') # TODO
flags.DEFINE_integer('image_height', default=32, help='')
flags.DEFINE_integer('image_width', default=32, help='')
flags.DEFINE_float('initial_lr', default=0.008, help='initial_lr when train batch size is 256')
flags.DEFINE_float('lr_decay', default=0.97, help='should be adjusted according to data size, and initial_lr') # TODO
flags.DEFINE_integer('num_train_images', default=50000, help='cifar train images')
flags.DEFINE_integer('num_test_images', default=10000, help='cifar test images')
# networks
flags.DEFINE_string('network', default='resnet20', help='network type')
# operations
flags.DEFINE_integer('num_threads', default=4, help='number of CPU threads for data processing')
flags.DEFINE_string('gpu', default='0', help='comma separated list of GPU(s)')
flags.DEFINE_string('test', default=None, help='run in test mode')
flags.DEFINE_string('test_dir', default=None, help='path to folder containing images to test')
flags.DEFINE_string('load', default=None, help='load a pre-trained model (for resuming or testing)')


class Cifar100(object):
    '''
    CIFAR100 (32x32, 100 classes, 20 super-calsses, 500 images/class (train), 100 images/class (test))
    https://www.cs.toronto.edu/~kriz/cifar.html

    Superclass 	                    Classes
    aquatic mammals 	            beaver, dolphin, otter, seal, whale
    fish 	                        aquarium fish, flatfish, ray, shark, trout
    flowers 	                    orchids, poppies, roses, sunflowers, tulips
    food containers 	            bottles, bowls, cans, cups, plates
    fruit and vegetables 	        apples, mushrooms, oranges, pears, sweet peppers
    household electrical devices 	clock, computer keyboard, lamp, telephone, television
    household furniture 	        bed, chair, couch, table, wardrobe
    insects 	                    bee, beetle, butterfly, caterpillar, cockroach
    large carnivores 	            bear, leopard, lion, tiger, wolf
    large man-made outdoor things 	bridge, castle, house, road, skyscraper
    large natural outdoor scenes 	cloud, forest, mountain, plain, sea
    large omnivores and herbivores 	camel, cattle, chimpanzee, elephant, kangaroo
    medium-sized mammals 	        fox, porcupine, possum, raccoon, skunk
    non-insect invertebrates 	    crab, lobster, snail, spider, worm
    people 	                        baby, boy, girl, man, woman
    reptiles 	                    crocodile, dinosaur, lizard, snake, turtle
    small mammals 	                hamster, mouse, rabbit, shrew, squirrel
    trees 	                        maple, oak, palm, pine, willow
    vehicles 1 	                    bicycle, bus, motorcycle, pickup truck, train
    vehicles 2 	                    lawn-mower, rocket, streetcar, tank, tractor
    '''
    def __init__(self, data_dir, valid_ids=None, label_map=None):
        if data_dir is None:
            return

        def load(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                data = dict[b'data']
                coarse_labels = np.asarray(dict[b'coarse_labels'])
                fine_labels = np.asarray(dict[b'fine_labels'])
                images = data.reshape([-1, 3, FLAGS.image_height * FLAGS.image_width])
                images = np.transpose(images, axes=[0, 2, 1])
                images = images.reshape([-1, FLAGS.image_height, FLAGS.image_width, 3])
                coarse_labels = coarse_labels.reshape([-1, 1])
                fine_labels = fine_labels.reshape([-1, 1])
                return images, fine_labels

        self.train_images, self.train_labels = load(os.path.join(data_dir, 'train'))
        # select a subset
        if valid_ids is not None:
            self.train_images = self.train_images[valid_ids]
            self.train_labels = self.train_labels[valid_ids]
        print('train images (numpy): {}'.format(self.train_images.shape))
        print('train labels (numpy): {}'.format(self.train_labels.shape))            

        self.test_images, self.test_labels = load(os.path.join(data_dir, 'test'))
        print('test images (numpy): {}'.format(self.test_images.shape))
        print('test labels (numpy): {}'.format(self.test_labels.shape))

        # use new labels
        if label_map is not None:            
            for i in range(self.train_labels.shape[0]):
                self.train_labels[i] = label_map[self.train_labels[i]]
            for i in range(self.test_labels.shape[0]):
                self.test_labels[i] = label_map[self.test_labels[i]]
            print('updated labels with the provided label map.')
            train_filter = list()
            for i in range(self.train_images.shape[0]):
                if self.train_labels[i] > -1:
                    train_filter.append(i)               
            self.train_images = self.train_images[train_filter]
            self.train_labels = self.train_labels[train_filter]
            FLAGS.num_train_images = self.train_images.shape[0]
            print('train images (numpy): {}'.format(self.train_images.shape))
            print('train labels (numpy): {}'.format(self.train_labels.shape))                        
            test_filter = list()
            for i in range(self.test_images.shape[0]):
                if self.test_labels[i] > -1:
                    test_filter.append(i)
            self.test_images = self.test_images[test_filter]   
            self.test_labels = self.test_labels[test_filter]         
            FLAGS.num_test_images = self.test_images.shape[0]
            print('test images (numpy): {}'.format(self.test_images.shape))
            print('test labels (numpy): {}'.format(self.test_labels.shape))


    def get_train_data(self):
        for i in range(self.train_images.shape[0]):
            yield self.train_images[i], self.train_labels[i]

    def get_test_data(self):
        for i in range(self.test_images.shape[0]):
            yield self.test_images[i], self.test_labels[i]

    def get_mean(self):
        return np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)

    def get_stddev(self):
        return np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    def process_train_data(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, 38, 38)
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, 63./255)
        # image = tf.image.random_contrast(image, 0.2, 1.8)
        # image = tf.image.resize(image, (FLAGS.image_height, FLAGS.image_width), 
        #     method=tf.image.ResizeMethod.BICUBIC)
        # image = tf.image.per_image_standardization(image)
        image -= tf.constant(self.get_mean())
        image /= tf.constant(self.get_stddev())
        return image, label

    def process_test_data(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        # image = tf.image.crop_to_bounding_box(image, 1, 1, 30, 30) # center crop
        # image = tf.image.resize(image, (FLAGS.image_height, FLAGS.image_width), 
        #     method=tf.image.ResizeMethod.BICUBIC)
        # image = tf.image.per_image_standardization(image)
        image -= tf.constant(self.get_mean())
        image /= tf.constant(self.get_stddev())
        return image, label

class Cifar10(object):
    '''
    CIFAR10 (32x32, 10 classes, 50K train, 10K test)
    0 = airplane, 1 = automobile, 2 = bird, 3 = cat, 4 = deer
    5 = dog, 6 = frog, 7 = horse, 8 = ship, 9 = truck
    '''
    def __init__(self, data_dir, valid_ids=None):
        if data_dir is None:
            return

        def load(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                data = dict[b'data']
                labels = np.asarray(dict[b'labels'])
                images = data.reshape([-1, 3, FLAGS.image_height*FLAGS.image_width])
                images = np.transpose(images, axes=[0, 2, 1])
                images = images.reshape([-1, FLAGS.image_height, FLAGS.image_height, 3])
                labels = labels.reshape([-1, 1])
                return images, labels

        # Read 5 training batches
        self.train_images, self.train_labels = load(os.path.join(data_dir, 'data_batch_1'))
        for i in range(2, 6):
            images, labels = load(os.path.join(data_dir, 'data_batch_{}'.format(i)))
            self.train_images = np.vstack((self.train_images, images))
            self.train_labels = np.vstack((self.train_labels, labels))

        # select a subset
        if valid_ids is not None:
            self.train_images = self.train_images[valid_ids]
            self.train_labels = self.train_labels[valid_ids]
        print('train images (numpy): {}'.format(self.train_images.shape))
        print('train labels (numpy): {}'.format(self.train_labels.shape))

        # Read 1 test batch
        self.test_images, self.test_labels = load(os.path.join(data_dir, 'test_batch'))
        print('test images (numpy): {}'.format(self.test_images.shape))
        print('test labels (numpy): {}'.format(self.test_labels.shape))

    def get_train_data(self):
        for i in range(self.train_images.shape[0]):
            yield self.train_images[i], self.train_labels[i]

    def get_test_data(self):
        for i in range(self.test_images.shape[0]):
            yield self.test_images[i], self.test_labels[i]

    def get_mean(self):
        return np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)

    def get_stddev(self):
        return np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    def process_train_data(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, 38, 38)
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, 63./255)
        # image = tf.image.random_contrast(image, 0.2, 1.8)
        # image = tf.image.resize(image, (FLAGS.image_height, FLAGS.image_width), 
        #     method=tf.image.ResizeMethod.BICUBIC)
        # image = tf.image.per_image_standardization(image)
        image -= tf.constant(self.get_mean())
        image /= tf.constant(self.get_stddev())
        return image, label

    def process_test_data(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        # image = tf.image.crop_to_bounding_box(image, 1, 1, 30, 30) # center crop
        # image = tf.image.resize(image, (FLAGS.image_height, FLAGS.image_width), 
        #     method=tf.image.ResizeMethod.BICUBIC)
        # image = tf.image.per_image_standardization(image)
        image -= tf.constant(self.get_mean())
        image /= tf.constant(self.get_stddev())
        return image, label


class CifarModel(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def load_model(self, model_path):
        '''
        Load a compiled model
        '''            
        with self.strategy.scope():
            model = keras.models.load_model(model_path, compile=True)
            model.summary()            
            return model

    def build_model(self, name, params):        
        '''
        params : dict for describing model parameters
        '''
        assert name in ['alexnet', 'vgg16', 'resnet20', 'resnet56', 'effnet0']

        with self.strategy.scope():
            input_shape = [FLAGS.image_height, FLAGS.image_width, 3]
            if FLAGS.channels_first:
                input_shape = [3, FLAGS.image_height, FLAGS.image_width]
            if name == 'alexnet':
                model = build_alexnet(input_shape, num_classes=FLAGS.num_classes)
            elif name == 'vgg16':
                model = build_vgg16(input_shape, num_classes=FLAGS.num_classes)
            elif name == 'resnet20':
                model = build_resnet(input_shape, n=2, num_classes=FLAGS.num_classes)
            elif name == 'resnet56':
                model = build_resnet(input_shape, n=6, num_classes=FLAGS.num_classes)
            elif name == 'effnet0':
                model = EfficientNetB0(params).get_model(input_shape, no_strides_at_first=True)
            model.summary()
            return model

class CifarTrainer(object):
    def __init__(self, strategy, data, model, logdir, compiled):
        '''
        strategy : distributed training strategy
        data : data handler
        model : keras model
        logdir : logdir for tensorboard
        compiled : the model is compiled or not
        '''
        self.strategy = strategy
        self.data = data
        self.model = model
        self.logdir = logdir
        self.compiled = compiled    

    def _build_dataset(self, train_or_eval):        
        assert train_or_eval in ['train', 'eval']
        if train_or_eval == 'train':
            ds = tf.data.Dataset.from_generator(self.data.get_train_data, 
                (tf.uint8, tf.int64), ([FLAGS.image_height, FLAGS.image_width, 3], 1))                
            ds = ds.shuffle(buffer_size=FLAGS.batch_size_per_replica*10)
            ds = ds.map(lambda image, label: self.data.process_train_data(image, label), 
                        num_parallel_calls=FLAGS.num_threads)
            ds = ds.batch(FLAGS.batch_size_per_replica * self.strategy.num_replicas_in_sync,
                          drop_remainder=True)
        else:
            ds = tf.data.Dataset.from_generator(self.data.get_test_data, 
                (tf.uint8, tf.int64), ([FLAGS.image_height, FLAGS.image_width, 3], 1))                
            ds = ds.map(lambda image, label: self.data.process_test_data(image, label), 
                        num_parallel_calls=FLAGS.num_threads)
            ds = ds.batch(min(FLAGS.num_test_images//100, 256), drop_remainder=False)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def _get_loss(self):        
        '''
        For using multiple devices, losses 
         must be called within strategy scope.
        '''
        # weight decay on all dense layers
        for layer in self.model.layers:
            #if isinstance(layer, tf.keras.layers.Dense) and hasattr(layer, 'kernel_regularizer'):
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = tf.keras.regularizers.l2(FLAGS.weight_decay)
        return keras.losses.SparseCategoricalCrossentropy()
    
    def _get_optimizer(self):        
        '''
        For using multiple devices, optimizers 
         must be called within strategy scope.
        '''        
        # The learning_rate is overwritten at the beginning of each step by callback
        # return keras.optimizers.RMSprop(self._lr_schedule(0), rho=0.9, momentum=0.9, epsilon=0.001)
        # return keras.optimizers.SGD(self._lr_schedule(0), momentum=0.9, nesterov=True)
        return keras.optimizers.Adam(self._lr_schedule(0))

    def _lr_schedule(self, epoch):
        train_batch_size = FLAGS.batch_size_per_replica * self.strategy.num_replicas_in_sync
        scaled_lr = FLAGS.initial_lr * (train_batch_size / 256)
        lr = scaled_lr * (FLAGS.lr_decay**epoch)
        # if epoch > 180:
        #     lr = scaled_lr * 0.0005
        # elif epoch > 160:
        #     lr = scaled_lr * 0.001
        # elif epoch > 120:
        #     lr = scaled_lr * 0.01
        # elif epoch > 80:
        #     lr = scaled_lr * 0.1
        # else:
        #     lr = scaled_lr
        # tf.summary.scalar('learning_rate', lr)
        return lr
    
    def train(self):
        writer = tf.summary.create_file_writer(self.logdir)
        writer.set_as_default()

        total_batch_size = FLAGS.batch_size_per_replica * self.strategy.num_replicas_in_sync

        if self.compiled is not True:
            with self.strategy.scope():
                self.model.compile(optimizer=self._get_optimizer(), 
                                   loss=self._get_loss(),
                                   metrics=[keras.metrics.SparseCategoricalAccuracy()])

        class DebugCallback(tf.keras.callbacks.Callback):            
            def __init__(self, strategy):
                super().__init__()
                self.strategy = strategy

            def on_train_batch_begin(self, batch, logs=None):
                if batch%100 == 0 and batch > 0:
                    self.btime = time.time()

            def on_train_batch_end(self, batch, logs=None):
                if batch%100 == 0 and batch > 0:
                    delta = time.time()-self.btime
                    images_per_sec = total_batch_size / delta
                    print('Training: batch {}, loss {:.4f} in {:.4f} images/sec'.format(
                        batch, logs['loss'], images_per_sec))        

        callbacks = [
            keras.callbacks.TensorBoard(self.logdir),
            keras.callbacks.LearningRateScheduler(self._lr_schedule, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                factor=0.31, patience=5, min_lr=0.5e-6, verbose=1),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.logdir, 'model-{epoch:05d}.h5'),
                save_best_only=True,
                monitor='val_loss',
                verbose=1),
            # DebugCallback(self.strategy),
        ]

        self.model.fit(self._build_dataset('train'),
            epochs=FLAGS.epochs,
            shuffle=True,
            verbose=2, # 0=silent, 1=progress bar, 2=one line per epoch
            validation_data=self._build_dataset('eval'),
            validation_steps=FLAGS.num_test_images//min(FLAGS.num_test_images//100, 256), # TODO
            callbacks=callbacks)

class CifarTester(object):
    def __init__(self, filepath):
        self.model = keras.models.load_model(filepath)
        self.model.summary()

    def predict(self, images):
        '''
        images : numpy array containing float32 RGB images
                 normalized properly by CIFAR statistics
        '''
        labels = self.model.predict(images, verbose=1)
        return labels

    def layer_output(self, images, layer_names):
        '''
        images : numpy array containing float32 RGB images
                 normalized properly by CIFAR statistics
        '''
        # func = keras.backend.function([self.model.layers[0].input, keras.backend.learning_phase()], 
        #                               [self.model.get_layer(layer_name).output])
        # output = func([images, 0])[0]
        intermediate_layer_model = keras.models.Model(inputs=self.model.input,
            outputs=[self.model.get_layer(l).output for l in layer_names])
        output = intermediate_layer_model.predict(images, verbose=1)
        return output

def main(argv):
    if FLAGS.gpu is not None:
        #os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

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

    if FLAGS.channels_first:
        tf.keras.backend.set_image_data_format('channels_first')

    if FLAGS.test:
        if FLAGS.num_classes == 10:
            data = Cifar10(data_dir=None)
        elif FLAGS.num_classes == 100:
            data = Cifar100(data_dir=None)
        else:
            print('Invalid num_classes (should be 10 or 100)')
            exit(0)
        mean = data.get_mean()
        stddev = data.get_stddev()
        images = list()
        fnames = list()
        for root, dirs, files in os.walk(FLAGS.test_dir):
            for file in tqdm(files):
                if file.endswith(tuple(['.jpg', '.jpeg', '.png', '.bmp'])):
                    try:   
                        img = io.imread(os.path.join(root, file))
                        img = img_as_float32(img)
                        img -= mean                      
                        img /= stddev
                        images.append(img)
                        fnames.append(file)
                    except Exception as e:
                        print('Error in reading', file, ':', e)        

        tester = CifarTester(FLAGS.load)
        # labels = tester.predict(np.array(images))
        feats = tester.layer_output(np.array(images), ['flatten', 'dense']) # resnet56
        # feats = tester.layer_output(np.array(images), ['flatten', 'dense_2']) # alexnet
        np.savez('feats.npz', fnames, feats[0], feats[1])
    else:
        valid_train_ids = np.load('/home/minje/dev/jnote/valid_train_ids.npy')
        label_map = np.load('/home/minje/dev/jnote/label_map.npy')
        if FLAGS.num_classes == 10:
            data = Cifar10('/home/minje/dev/dataset/cifar/cifar-10-batches-py/', 
                valid_ids=valid_train_ids) # no label map is used for cifar10
        elif FLAGS.num_classes == 100:
            data = Cifar100('/home/minje/dev/dataset/cifar/cifar100/cifar-100-python/', 
                valid_ids=valid_train_ids,
                label_map=label_map)
        if FLAGS.load is None:
            params = dict({
                'num_classes' : FLAGS.num_classes,
                'MB_blocks' : {
                        'repeats' : [1, 2, 2, 3, 3, 4, 1],
                        'in_channels' : [32, 16, 24, 40, 80, 112, 192],
                        'expand_ratio' : [1, 6, 6, 6, 6, 6, 6],
                        'kernel_sizes' : [3, 3, 5, 3, 5, 5, 3],
                        'strides' : [1, 2, 2, 2, 1, 2, 1], # differnt from paper (comply with official implementation)
                        'output_filters' : [16, 24, 40, 80, 112, 192, 320],
                        'id_skip' : [True, True, True, True, True, True, True],
                    },
                'drop_connect_rate' : 0.2,
                'dropout_rate' : 0.2,
            })
            model = CifarModel(strategy).build_model(FLAGS.network, params)
            compiled = False
        else:
            model = CifarModel(strategy).load_model(FLAGS.load)
            compiled = True
        fname = os.path.basename(sys.argv[0]) # argv[0] contains python filename
        fname = fname[:fname.rfind('.')]
        logdir = os.path.join('train_log/', fname)
        stamp = time.strftime('%Y%B%d-%H%M%S', time.gmtime())
        logdir = os.path.join(logdir, stamp)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        trainer = CifarTrainer(strategy, data, model, logdir, compiled)
        trainer.train()

if __name__ == '__main__':
    app.run(main)