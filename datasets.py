from __future__ import print_function
from __future__ import division

import os
import glob
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
import config

img_size = config.img_size
classes = config.classes
data_path = config.data_path
train_list_file = config.train_list_file
trainval_list_file = config.trainval_list_file
suffix_name = config.suffix_name

train_filelist = os.path.join(data_path, train_list_file)
trainval_filelist = os.path.join(data_path, trainval_list_file)

def load_train(train_path, anno_path , image_size=512, mode='train'):

    img_names = []
    anno_names = []

    basenames = []

    if mode == 'train':
        f = open(train_filelist, 'r')
    elif mode == 'trainval':
        f = open(trainval_filelist)

    filenames = f.readlines()
    for filename in filenames:
        basenames.append(filename.strip() + suffix_name)

    index = 0
    for img_name in basenames:
        img_name = os.path.join(train_path, img_name)
        img_names.append(img_name)
        #print('%d: %s is ok' %(index, img_name))
        index += 1



    if mode == 'trainval':
        return img_names, anno_names

    index = 0
    for anno_name in basenames:
        anno_name = os.path.join(anno_path, anno_name)
        anno_names.append(anno_name)
        #print('%d: %s is ok' % (index, anno_name))
        index += 1




    return img_names, anno_names

class DataSet(object):
    def __init__(self, img_names, anno_names):
        self._num_examples = len(img_names)
        self._img_names = img_names
        self._anno_names = anno_names
        self._epochs_done = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        images = []
        annos = []

        for i in range(start, end):
            img_name = self._img_names[i]
            img = Image.open(img_name)
            img = np.array(img)
            img = img.astype(np.float32)
            img = np.multiply(img, 1.0 / 255.0)
            images.append(img)

            if len(self._anno_names) == 0:
                labels = np.zeros_like(img, dtype=np.int32)
                annos.append(labels)
            else:
                anno_name = self._anno_names[i]
                img = Image.open(anno_name)
                img = np.array(img)
                labels = img.astype(np.int32)
                annos.append(labels)

        images = np.array(images)
        annos = np.array(annos)



        return images, annos, self._img_names[start:end], self._anno_names[start:end]


def read_data_sets(img_path, anno_path, image_size=512, mode='train'):
    class DataSets(object):
        pass
    data_sets = DataSets()

    img_names, anno_names = load_train(img_path, anno_path, image_size, mode=mode)

    if mode == 'train':
        img_names, anno_names = shuffle(img_names, anno_names)
        print("shuffle is ok")
    print("the number of the datasets is %d" % len(img_names))

    data_sets.data = DataSet(img_names, anno_names)

    print("train sets or test sets is ok")

    return data_sets
