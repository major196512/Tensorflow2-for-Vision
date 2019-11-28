import os
import pickle
import numpy as np
import yaml

import tensorflow as tf

class CIFAR10:
    def __init__(self, cfg, mode='train'):
        cfg = yaml.load(open(cfg))

        if mode == 'train' : self.train_dataset(cfg)
        elif mode == 'test' : self. test_dataset(cfg)
        else : raise ValueError('Invalide cifar-10 dataset mode')

    def train_dataset(self, cfg):
        self.img = np.zeros((0, cfg['img_size']))
        self.label = np.zeros((0, ))

        for i in range(1, cfg['num_train_file']+1):
            file_name = os.path.join(cfg['data_path'], cfg['train_file_name']+str(i))
            dict = pickle.load(open(file_name, 'rb'), encoding='bytes')

            img = np.array(dict[b'data'])
            label = np.array(dict[b'labels'])

            self.img = np.concatenate([self.img, img], axis=0)
            self.label = np.concatenate([self.label, label], axis=0)

    def test_dataset(self, cfg):
        file_name = os.path.join(cfg['data_path'], cfg['test_file_name'])
        dict = pickle.load(open(file_name, 'rb'), encoding='bytes')

        self.img = np.array(dict[b'data'])
        self.label = np.array(dict[b'labels'])

    #####################################################################
    def __call__(self):
        for i in range(self.img.shape[0]):
            yield (self.img[i], self.label[i])

    #####################################################################
    def __len__(self):
        return self.img.shape[0]
