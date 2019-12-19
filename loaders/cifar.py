import os
import pickle
import numpy as np
import yaml
from skimage import transform

import tensorflow as tf

toy_train = 1000
toy_test = 200

class CIFAR10:
    def __init__(self, cfg, mode='train'):
        data_cfg = cfg['DATA']
        loader_cfg = cfg['LOADER']
        self.channel_first = data_cfg['channel_first']

        if mode == 'train' : self.train_dataset(data_cfg)
        elif mode == 'val' : self.test_dataset(data_cfg)
        elif mode == 'test' : self.test_dataset(data_cfg)
        else : raise ValueError('Invalide cifar-10 dataset mode')

        self.num_img = self.img.shape[0]
        self.img = self.img.reshape(self.num_img, 3, 32, 32)
        if self.channel_first is False:
            self.img = np.transpose(self.img, (0, 2, 3, 1))

        self.normalizer(loader_cfg)

    #####################################################################
    def train_dataset(self, data_cfg):
        self.img = np.zeros((0, 3072))
        self.label = np.zeros((0, ))

        for i in range(5):
            file_name = os.path.join(data_cfg['data_dir'], 'data_batch_'+str(i+1))
            dict = pickle.load(open(file_name, 'rb'), encoding='bytes')

            img = np.array(dict[b'data'])
            label = np.array(dict[b'labels'])

            self.img = np.concatenate([self.img, img], axis=0)
            self.label = np.concatenate([self.label, label], axis=0)

        if data_cfg['toy'] : self.img = self.img[:toy_train]

    def test_dataset(self, data_cfg):
        file_name = os.path.join(data_cfg['data_dir'], 'test_batch')
        dict = pickle.load(open(file_name, 'rb'), encoding='bytes')

        self.img = np.array(dict[b'data'])
        self.label = np.array(dict[b'labels'])

        if data_cfg['toy'] : self.img = self.img[:toy_test]

    #####################################################################
    def normalizer(self, dataset_cfg):
        max_value = dataset_cfg['normalizer']['max_value']
        mean = dataset_cfg['normalizer']['mean']
        std = dataset_cfg['normalizer']['std']

        assert len(mean) == len(std)
        num_channel = len(mean)

        mean = np.array(mean).astype(np.float32)
        std = np.array(std).astype(np.float32)

        if self.channel_first:
            mean = mean.reshape(1, num_channel, 1, 1)
            std = std.reshape(1, num_channel, 1, 1)
        else:
            mean = mean.reshape(1, 1, 1, num_channel)
            std = std.reshape(1, 1, 1, num_channel)

        self.img = (self.img.astype(np.float32) / max_value - mean) / std

    #####################################################################
    def __call__(self):
        for i in range(self.num_img):
            yield (self.img[i], self.label[i])

    #####################################################################
    def __len__(self):
        return self.num_img
