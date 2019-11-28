from loaders.cifar import CIFAR10

def load_data(cfg):
    dataset = cfg['DATASET']
    if dataset['data'] == 'cifar_10':
        train_data = CIFAR10(dataset, mode='train')
        test_data = CIFAR10(dataset, mode='test')

    else:
        raise ValueError('Invalide data class')

    return train_data, test_data
