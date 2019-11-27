import os
import pickle
import numpy as np

import tensorflow as tf

class CIFAR10:
    def __init__(self, path, mode='train'):
        ############################################
        num_train_file = 5
        train_file_name = 'data_batch_'
        test_file_name = 'test_batch'
        data_key = b'data'
        label_key = b'labels'
        img_size = 3072

        ############################################
        self.train_img = np.zeros((0, img_size))
        self.test_img = np.zeros((0, img_size))
        self.train_label = np.zeros((0, ))
        self.test_label = np.zeros((0, ))

        ############################################
        for i in range(1, num_train_file+1):
            file_name = os.path.join(path, train_file_name+str(i))
            dict = pickle.load(open(file_name, 'rb'), encoding='bytes')

            img = np.array(dict[data_key])
            label = np.array(dict[label_key])

            self.train_img = np.concatenate([self.train_img, img], axis=0)
            self.train_label = np.concatenate([self.train_label, label], axis=0)

        ############################################
        file_name = os.path.join(path, test_file_name)
        dict = pickle.load(open(file_name, 'rb'), encoding='bytes')

        self.test_img = np.array(dict[data_key])
        self.test_label = np.array(dict[label_key])

        self.num_train = self.train_img.shape[0]
        self.num_test = self.test_img.shape[0]

    #####################################################################
    def __call__(self):
        for i in range(self.num_train):
            yield (self.train_img[i], self.train_label[i])
