import tensorflow as tf
#from tensorflow.keras.applications import ResNet50
from modules.ResNet import resnet

def load_model(cfg):
    #model = ResNet50()
    model = resnet(50, pretrained=False, channels_first=cfg['DATA']['channel_first'])
    sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
