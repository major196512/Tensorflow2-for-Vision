import tensorflow as tf
#from tensorflow.keras.applications import ResNet50
from modules.ResNet import resnet
# from modules.test import resnet

def load_model(cfg, pretrain=False):
    # model = resnet(50)
    model = resnet(50, pretrained=pretrain, channels_first=cfg['DATA']['channel_first'])
    # sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0001)
    sgd = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
