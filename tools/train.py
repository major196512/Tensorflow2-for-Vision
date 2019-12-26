import tensorflow as tf

import sys
import os
import yaml

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import merge_yaml
from utils.dataset import load_data
from utils.callback import Callback
from utils.model import load_model

def main():
    cfg = yaml.load(open('./configs/test_cifar.yaml'), Loader=yaml.FullLoader)
    cfg = merge_yaml(cfg)

    train_data, test_data = load_data(cfg)
    model = load_model(cfg, pretrain=True)
    callbacks = Callback(cfg)

    #train the model for the first time
    model.fit_generator(train_data,
              epochs=cfg['SOLVER']['epoch'],
              callbacks=callbacks,
              validation_data=test_data,
              validation_freq=1,
              shuffle=cfg['DATA']['shuffle']
              )
    a = model.evaluate_generator(test_data)
    print('Test Data Accuracy : %.2f' % a[1])

if __name__ == '__main__':
    main()
