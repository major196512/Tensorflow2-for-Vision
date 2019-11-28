import tensorflow as tf
import sys
import os
import yaml

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.merge_yaml import merge_yaml
from utils.load_data import load_data

def main():
    cfg = yaml.load(open('./configs/base.yaml'))
    additional_cfg = yaml.load(open('./configs/test_cifar.yaml'))

    cfg = merge_yaml(cfg, additional_cfg)

    train_data, test_data = load_data(cfg)
    num_train = len(train_data)
    num_test = len(test_data)

    train_loader = tf.data.Dataset.from_generator(train_data, (tf.int32, tf.int32))
    test_loader = tf.data.Dataset.from_generator(test_data, (tf.int32, tf.int32))

    train_loader = train_loader.shuffle(num_train).batch(cfg['SOLVER']['batch_size'], drop_remainder=True)
    for d, l in train_loader:
        print(l)

if __name__ == '__main__':
    main()
