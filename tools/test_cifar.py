import tensorflow as tf
import sys
import os

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.cifar import CIFAR10

def main():
    t = CIFAR10('/home/taeho/data/cifar-10-batches-py')
    ds = tf.data.Dataset.from_generator(t, (tf.int32, tf.int32))
    ds = ds.shuffle(50000).batch(64, drop_remainder=True)
    for d, l in ds:
        print(l)

if __name__ == '__main__':
    main()
