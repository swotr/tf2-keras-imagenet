from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# The input tensor is in the range of [0, 255], we need to scale them to the range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
CROP_PADDING = 32

def _parse_function(serialized):
    image_feature_desc = {
        'image/channels' : tf.io.FixedLenFeature([], tf.int64),
        'image/class/label' : tf.io.FixedLenFeature([], tf.int64),
        'image/class/synset' : tf.io.FixedLenFeature([], tf.string),
        'image/class/text' : tf.io.FixedLenFeature([], tf.string),
        'image/colorspace' : tf.io.FixedLenFeature([], tf.string),
        'image/encoded' : tf.io.FixedLenFeature([], tf.string),
        'image/filename' : tf.io.FixedLenFeature([], tf.string),
        'image/format' : tf.io.FixedLenFeature([], tf.string),
        'image/height' : tf.io.FixedLenFeature([], tf.int64),
        #'image/object/bbox/label' : tf.io.FixedLenFeature([], tf.int64),
        #'image/object/bbox/xmax' : tf.io.FixedLenFeature([], tf.float32),
        #'image/object/bbox/xmin' : tf.io.FixedLenFeature([], tf.float32),
        #'image/object/bbox/ymax' : tf.io.FixedLenFeature([], tf.float32),
        #'image/object/bbox/ymin' : tf.io.FixedLenFeature([], tf.float32),
        'image/width' : tf.io.FixedLenFeature([], tf.int64),
    }
    data = tf.io.parse_single_example(serialized, image_feature_desc)
    image_bytes = data['image/encoded']
    label = tf.cast(data['image/class/label'], dtype=tf.int32)
    return image_bytes, label

def _decode_and_center_crop(image_buffer, image_size):
    shape = tf.io.extract_jpeg_shape(image_buffer)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_height = tf.cast(
        ((image_size[0] / (image_size[0] + CROP_PADDING)) *
        tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)
    padded_center_crop_width = tf.cast(
        ((image_size[1] / (image_size[1] + CROP_PADDING)) *
        tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_height) + 1) // 2
    offset_width = ((image_width - padded_center_crop_width) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
        padded_center_crop_height, padded_center_crop_width])

    image = tf.io.decode_and_crop_jpeg(image_buffer, crop_window, channels=3)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BICUBIC)
    return image

def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)

def _decode_and_random_crop(image_buffer, image_size):
    shape = tf.io.extract_jpeg_shape(image_buffer)

    sample_distorted_bbox = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
        min_object_covered=0.1,
        aspect_ratio_range=(3./4, 4./3),
        area_range=(0.08, 1.0),
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)

    bbox_begin, bbox_size, _ = sample_distorted_bbox
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_h, target_w, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_h, target_w])
    image = tf.io.decode_and_crop_jpeg(image_buffer, crop_window, channels=3)

    bad = _at_least_x_are_equal(shape, tf.shape(image), 3)    
    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_buffer, image_size),
        lambda: tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BICUBIC))

    return image

def _normalize_image_and_label(image, label):
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    label = label - 1 # [1, 1000] --> [0, 999]
    return image, label

def preprocess_for_train(dataset, image_size):
    image_buffer, label = _parse_function(dataset)
    image = _decode_and_random_crop(image_buffer, image_size)
    image = tf.image.random_flip_left_right(image)
    image, label = _normalize_image_and_label(image, label)
    return image, label    

def preprocess_for_eval(dataset, image_size):
    image_buffer, label = _parse_function(dataset)
    image = _decode_and_center_crop(image_buffer, image_size)
    image, label = _normalize_image_and_label(image, label)
    return image, label