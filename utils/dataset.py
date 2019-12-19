import tensorflow as tf

from loaders.cifar import CIFAR10

def cifar10(cfg):
    train_mode = cfg['DATA']['train_set']
    test_mode = cfg['DATA']['test_set']

    train_data = CIFAR10(cfg, mode=train_mode)
    test_data = CIFAR10(cfg, mode=test_mode)

    num_train = len(train_data)
    num_test = len(test_data)

    train_data = tf.data.Dataset.from_generator(train_data, (tf.float32, tf.int32))
    test_data = tf.data.Dataset.from_generator(test_data, (tf.float32, tf.int32))

    return train_data, test_data

def load_data(cfg):
    dataset = cfg['DATA']['dataset']
    batch_size = cfg['SOLVER']['batch_size']

    if dataset == 'cifar_10' : loader = cifar10(cfg)
    else : raise ValueError('Invalide data class')

    train_data, test_data = loader

    train_data = train_data.batch(batch_size, drop_remainder=True)
    test_data = test_data.batch(batch_size, drop_remainder=True)

    return train_data, test_data
