import os
import pickle
import numpy as np
import yaml

import tensorflow as tf

class CIFAR10:
    def __init__(self, dataset_cfg, mode='train'):
        data_cfg = yaml.load(open(dataset_cfg['data_dir']), Loader=yaml.FullLoader)

        if mode == 'train' : self.train_dataset(data_cfg)
        elif mode == 'test' : self. test_dataset(data_cfg)
        else : raise ValueError('Invalide cifar-10 dataset mode')

        self.img = self.img.reshape(self.img.shape[0], data_cfg['channel'], data_cfg['width'], data_cfg['height'])
        self.normalizer(dataset_cfg)

    #####################################################################
    def train_dataset(self, data_cfg):
        self.img = np.zeros((0, data_cfg['img_size']))
        self.label = np.zeros((0, ))

        for i in range(1, data_cfg['num_train_file']+1):
            file_name = os.path.join(data_cfg['data_path'], data_cfg['train_file_name']+str(i))
            dict = pickle.load(open(file_name, 'rb'), encoding='bytes')

            img = np.array(dict[b'data'])
            label = np.array(dict[b'labels'])

            self.img = np.concatenate([self.img, img], axis=0)
            self.label = np.concatenate([self.label, label], axis=0)

    def test_dataset(self, data_cfg):
        file_name = os.path.join(data_cfg['data_path'], data_cfg['test_file_name'])
        dict = pickle.load(open(file_name, 'rb'), encoding='bytes')

        self.img = np.array(dict[b'data'])
        self.label = np.array(dict[b'labels'])

    #####################################################################
    def normalizer(self, dataset_cfg):
        max_value = dataset_cfg['normalizer']['max_value']
        mean = np.array([[dataset_cfg['normalizer']['mean']]]).astype(np.float32)
        std = np.array([[dataset_cfg['normalizer']['std']]]).astype(np.float32)

        mean = mean.reshape(len(dataset_cfg['normalizer']['mean']), 1, 1)
        std = std.reshape(len(dataset_cfg['normalizer']['std']), 1, 1)

        mean = np.expand_dims(mean, axis=0)
        std = np.expand_dims(std, axis=0)

        self.img = (self.img.astype(np.float32) / max_value - mean) / std

    #####################################################################
    def __call__(self):
        for i in range(self.img.shape[0]):
            yield (self.img[i], self.label[i])

    #####################################################################
    def __len__(self):
        return self.img.shape[0]
