#coding=utf-8

import numpy as np
import tensorflow as tf
import datasets
import os
import shutil

import config

import networks.unet

from numpy.random import seed
seed(10)

data_path = config.data_path
img_dir_name = config.img_dir_name
annotation_dir_name = config.annotation_dir_name

model = os.path.join(data_path, "model")

if os.path.exists(model):
	shutil.rmtree(model)
os.mkdir(model)

saved = os.path.join(data_path, "saved")

if os.path.exists(saved):
	shutil.rmtree(saved)
os.mkdir(saved)

batch_size = config.batch_size
img_size = config.img_size
num_channels = config.num_channels
classes = config.classes
learning_rate = config.learning_rate
weighted = config.weighted

max_steps = config.max_steps

train_path = os.path.join(data_path, img_dir_name)
anno_path = os.path.join(data_path, annotation_dir_name)

f = open(os.path.join(data_path, 'loss.txt'), 'w')

data = datasets.read_data_sets(train_path, anno_path, img_size)

def weighted_loss(logits, labels, num_classes, head=None):
    """re-weighting"""
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))

        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        label_flat = tf.reshape(labels, (-1, 1))
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss

def cal_loss(logits, labels):
    '''loss_weight = np.array([
        0.5,
        1.0,
        2.0,
        1.0
    ])
    '''
    loss_weight = np.array([
        0.5,
        1.0
    ])

    labels = tf.cast(labels, tf.int32)

    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=classes, head=loss_weight)


sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, None, None, num_channels], name='x')
#y_true = tf.placeholder(tf.float32, shape=[None, img_size, img_size, classes], name='y_true')  # how to transform anno to 4-dimension image
y_true = tf.placeholder(tf.int32, shape=[None, None, None], name='y_true')  # Sparse representation


y_pred = networks.unet.create_unet(x)

if weighted == "yes":
    cost_reg = cal_loss(y_pred, y_true)
else:
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    tf.add_to_collection("losses", cost)
    cost_reg = tf.add_n(tf.get_collection("losses"))


y_pred = tf.argmax(y_pred, axis = 3, name="y_pred")



tf.summary.scalar("loss", cost_reg)
summ = tf.summary.merge_all()
writer = tf.summary.FileWriter(os.path.join(data_path, 'summary'))
writer.add_graph(sess.graph)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_reg)

total_iterations = 0

saver = tf.train.Saver()

def show_progress(epoch, feed_dict_train, i, cost):
    loss = sess.run(cost, feed_dict=feed_dict_train)

    msg = "Traing Epoch{0} --- iterations: {1}  --- Training loss: {2}"
    print(msg.format(epoch + 1, i, loss))

def train(num_iteration):
    global total_iterations
    sess.run(tf.global_variables_initializer())

    for i in range(total_iterations, total_iterations + num_iteration + 1):
        x_batch, y_true_batch, img_names_bathc, anno_names_patch = data.data.next_batch(batch_size)
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        y_arr = sess.run(y_pred, feed_dict=feed_dict_tr)
        #print(np.max(y_arr))

        sess.run(optimizer, feed_dict=feed_dict_tr)
        loss = sess.run(cost_reg, feed_dict=feed_dict_tr)

        cont = str(np.max(y_arr)) + ": " + str(loss) + "\n"
        f.write(cont)

        epoch = int(i / int(data.data._num_examples / batch_size))
        show_progress(epoch, feed_dict_tr, i, cost_reg)

        if i % int(data.data._num_examples/batch_size) == 0 or i == total_iterations + num_iteration:
            saver.save(sess, os.path.join(data_path, 'model/segmentation.ckpt'), global_step=i)
            saved_np = os.path.join(saved, str(i) + ".npy")
            np.save(saved_np, y_arr)

    total_iterations += num_iteration

train(num_iteration = max_steps)


