import tensorflow as tf
import tensorflow.keras.layers as layers
import yaml
class Test(tf.keras.Model):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        #self.conv1 = layers.Dense(64)

    def call(self, input):
        x = self.conv1(input)

        return out

if __name__ == '__main__':
    model = Test()
    # model = tf.keras.models.Sequential()
    # model.add(layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False, name='asdf'))
    # print(model.to_yaml())
    model.save_weights('test')
