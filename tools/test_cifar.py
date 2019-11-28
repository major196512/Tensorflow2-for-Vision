import tensorflow as tf
import sys
import os
import yaml

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.cifar import CIFAR10

def main():
    cfg_file_name = './configs/test_cifar.yaml'
    cfg = yaml.load(open(cfg_file_name))

    train_data = CIFAR10(cfg['data_config'], mode='train')
    test_data = CIFAR10(cfg['data_config'], mode='test')

    num_train = len(train_data)
    num_test = len(test_data)

    train_loader = tf.data.Dataset.from_generator(train_data, (tf.int32, tf.int32))
    test_loader = tf.data.Dataset.from_generator(test_data, (tf.int32, tf.int32))

    train_loader = train_loader.shuffle(num_train).batch(cfg['batch_size'], drop_remainder=True)
    for d, l in train_loader:
        print(l)

if __name__ == '__main__':
    main()
