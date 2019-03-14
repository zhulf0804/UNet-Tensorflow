#coding=utf-8

import numpy as np
import tensorflow as tf
import config
import datetime

batch_size = config.batch_size
classes = config.classes
img_size = config.img_size


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def conv_layer(input, num_input_channels, conv_filter_size, num_filters, padding='SAME', relu=True):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=padding)
    layer += biases

    if relu:
        layer = tf.nn.relu(layer)
    return layer

def pool_layer(input, padding='SAME'):
    return tf.nn.max_pool(value=input,
                          ksize = [1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding=padding)

def un_conv(input, num_input_channels, conv_filter_size, num_filters, feature_map_size, train=True, padding='SAME',relu=True):


    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_filters, num_input_channels])
    biases = create_biases(num_filters)
    if train:
        batch_size_0 = batch_size
    else:
        batch_size_0 = 1
    layer = tf.nn.conv2d_transpose(value=input, filter=weights,
                                   output_shape=[batch_size_0, feature_map_size, feature_map_size, num_filters],
                                   strides=[1, 2, 2, 1],
                                   padding=padding)
    layer += biases

    if relu:
        layer = tf.nn.relu(layer)
    return layer


def create_unet(input, train=True):

    # train is used for un_conv, to determine the batch size

    conv1 = conv_layer(input, 3, 3, 64)
    conv2 = conv_layer(conv1, 64, 3, 64)
    pool2 = pool_layer(conv2)
    conv3 = conv_layer(pool2, 64, 3, 128)
    conv4 = conv_layer(conv3, 128, 3, 128)
    pool4 = pool_layer(conv4)
    conv5 = conv_layer(pool4, 128, 3, 256)
    conv6 = conv_layer(conv5, 256, 3, 256)
    pool6 = pool_layer(conv6)
    conv7 = conv_layer(pool6, 256, 3, 512)
    conv8 = conv_layer(conv7, 512, 3, 512)
    pool8 = pool_layer(conv8)

    conv9 = conv_layer(pool8, 512, 3, 1024)
    conv10 = conv_layer(conv9, 1024, 3, 1024)

    conv11 = un_conv(conv10, 1024, 2, 512, img_size // 8, train)
    merge11 = tf.concat(values=[conv8, conv11], axis = -1)

    conv12 = conv_layer(merge11, 1024, 3, 512)
    conv13 = conv_layer(conv12, 512, 3, 512)

    conv14 = un_conv(conv13, 512, 2, 256, img_size // 4, train)
    merge14 = tf.concat([conv6, conv14], axis=-1)

    conv15 = conv_layer(merge14, 512, 3, 256)
    conv16 = conv_layer(conv15, 256, 3, 256)

    conv17 = un_conv(conv16, 256, 2, 128, img_size // 2, train)
    merge17 = tf.concat([conv17, conv4], axis=-1)

    conv18 = conv_layer(merge17, 256, 3, 128)
    conv19 = conv_layer(conv18, 128, 3, 128)

    conv20 = un_conv(conv19, 128, 2, 64, img_size, train)
    merge20 = tf.concat([conv20, conv2], axis=-1)

    conv21 = conv_layer(merge20, 128, 3, 64)
    conv22 = conv_layer(conv21, 64, 3, 64)
    conv23 = conv_layer(conv22, 64, 1, classes)

    return conv23

if __name__ == '__main__':
    oldtime = datetime.datetime.now()
    for i in range(16):
        input = tf.constant(0.6, shape=[1, 512, 512, 3])
        y = create_unet(input, train=False)
        #print(y)
    newtime = datetime.datetime.now()

    print('the interval is：%s s' % (newtime - oldtime).seconds)
