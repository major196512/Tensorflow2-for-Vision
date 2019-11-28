from loaders.cifar import CIFAR10

def load_data(cfg):
    dataset = cfg['DATASET']
    if dataset['data'] == 'cifar_10':
        train_data = CIFAR10(dataset['data_dir'], mode='train')
        test_data = CIFAR10(dataset['data_dir'], mode='test')

    else:
        raise ValueError('Invalide data class')

    return train_data, test_data
