#coding=utf-8
from __future__ import  print_function

import tensorflow as tf
import numpy as np
import shutil
import os, glob
import sys, argparse
from PIL import Image

import config
import datasets

batch_size = config.batch_size
img_size = config.img_size
num_channels = config.num_channels
classes = config.classes
learning_rate = config.learning_rate
max_steps = config.max_steps

data_path = config.data_path
img_dir_name = config.img_dir_name
annotation_dir_name = config.annotation_dir_name
trainval_list_file = config.trainval_list_file


img_path = os.path.join(data_path, img_dir_name)
anno_path = os.path.join(data_path, annotation_dir_name)

saved_path = os.path.join(data_path, 'prediction_trainval')

if os.path.exists(saved_path):
	shutil.rmtree(saved_path)
os.mkdir(saved_path)

data = datasets.read_data_sets(img_path, anno_path, img_size, mode='trainval')


sess = tf.Session()

saver = tf.train.import_meta_graph(os.path.join(data_path, 'model/segmentation.ckpt-'+ str(max_steps) + '.meta'))  ##获取
saver.restore(sess, os.path.join(data_path, 'model/segmentation.ckpt-' + str(max_steps)))

graph = tf.get_default_graph()

##验证集效果不好,注意keep_prob，在测试时设置为1

y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 2))  #np.zeros

def predict(total_num):
    for i in range(total_num):
        x_batch, y_true_batch, img_names_batch, anno_names_patch = data.data.next_batch(1) # batchsize = 1 for test
        feed_dict_tr = {x: x_batch}
        results = sess.run(y_pred, feed_dict=feed_dict_tr)

        results.astype(np.uint8)

        print(np.max(results))

        results *= 60  # for visualization

        for j in range(batch_size):
            img = results[j, :, :]
            img = Image.fromarray(img)
            img_name = os.path.join(saved_path, os.path.basename(img_names_batch[j]))
            print(img_name)
            img.save(img_name)



if __name__ == '__main__':
    trainval_filelist = os.path.join(data_path, trainval_list_file)
    f = open(trainval_list_file, 'r')
    lines = f.readlines()
    predict(len(lines))













