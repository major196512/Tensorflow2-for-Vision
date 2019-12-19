import tensorflow as tf
import tensorflow.keras.layers as layers

class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, channel, stride=1, downsample=None, channels_first=True):
        super(BasicBlock, self).__init__()
        if channels_first :
            data_format = 'channels_first'
            axis=1
        else :
            data_format = 'channels_last'
            axis=3

        self.conv1 = layers.Conv2D(channel, kernel_size=3, strides=stride, padding='same', use_bias=False, data_format=data_format)
        self.bn1 = layers.BatchNormalization(axis=axis)
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(channel, kernel_size=3, padding='same', use_bias=False, data_format=data_format)
        self.bn2 = layers.BatchNormalization(axis=axis)
        self.relu2 = layers.ReLU()

        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)

        return out

class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, channel, stride=1, downsample=None, channels_first=True):
        super(Bottleneck, self).__init__()
        if channels_first :
            data_format = 'channels_first'
            axis=1
        else :
            data_format = 'channels_last'
            axis=3

        self.stride = stride
        self.conv1 = layers.Conv2D(channel, kernel_size=1, padding='same', use_bias=False, data_format=data_format)
        self.bn1 = layers.BatchNormalization(axis=axis)
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(channel, kernel_size=3, strides=stride, padding='same', use_bias=False, data_format=data_format)
        self.bn2 = layers.BatchNormalization(axis=axis)
        self.relu2 = layers.ReLU()

        self.conv3 = layers.Conv2D(channel*4, kernel_size=1, padding='same', use_bias=False, data_format=data_format)
        self.bn3 = layers.BatchNormalization(axis=axis)
        self.relu3 = layers.ReLU()

        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu3(out)

        return out

class ResNet(tf.keras.Model):
    def __init__(self, block, layer, channels_first=True):
        self.inplanes = 64

        super(ResNet, self).__init__()
        if channels_first :
            self.data_format = 'channels_first'
            self.axis=1
        else :
            self.data_format = 'channels_last'
            self.axis=3

        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False, data_format=self.data_format)
        self.bn1 = layers.BatchNormalization(axis=self.axis)
        self.relu1 = layers.ReLU()
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same', data_format=self.data_format)

        self.layer1 = self._make_layer(block, 64, layer[0])
        self.layer2 = self._make_layer(block, 128, layer[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer[3], stride=2)

        #self.avg_pool = layers.AveragePooling2D(pool_size, strides, padding, data_format=self.data_format)
        self.reshape = layers.Reshape((512 * block.expansion, ))
        self.fc = layers.Dense(10, use_bias=False)
        self.softmax = layers.Softmax()


        self.size =[64,
                    64 * block.expansion,
                    128 * block.expansion,
                    256 * block.expansion,
                    512 * block.expansion]

        self.init_params()

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != channel * block.expansion:
            downsample = tf.keras.Sequential()
            downsample.add(layers.Conv2D(channel * block.expansion, kernel_size=1, strides=stride, use_bias=False, data_format=self.data_format))
            downsample.add(layers.BatchNormalization(axis=self.axis))

        layer = tf.keras.Sequential()
        layer.add(block(channel, stride, downsample))
        for i in range(1, blocks):
            layer.add(block(channel))

        return layer

    def init_params(self):
        pass

    def freeze_modules(self, freeze):
        pass

    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        C1 = self.maxpool(x)

        C2 = self.layer1(C1)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        out = self.reshape(C5)
        out = self.fc(out)
        out = self.softmax(out)

        return out

    def get_size(self):
        return self.size

def resnet18(pretrained=False, channels_first=True):
    model = ResNet(BasicBlock, [2, 2, 2, 2], channels_first=channels_first)
    if pretrained:
        pass
    return model

def resnet34(pretrained=False, channels_first=True):
    model = ResNet(BasicBlock, [3, 4, 6, 3], channels_first=channels_first)
    if pretrained:
        pass
    return model

def resnet50(pretrained=False, channels_first=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], channels_first=channels_first)
    if pretrained:
        pass
    return model

def resnet101(pretrained=False, channels_first=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3], channels_first=channels_first)
    if pretrained:
        pass
    return model

def resnet152(pretrained=False, channels_first=True):
    model = ResNet(Bottleneck, [3, 8, 36, 3], channels_first=channels_first)
    if pretrained:
        pass
    return model

def resnet(net_size, pretrained=False, channels_first=True):
    if net_size == 18 : return resnet18(pretrained, channels_first)
    elif net_size == 34 : return resnet34(pretrained, channels_first)
    elif net_size == 50 : return resnet50(pretrained, channels_first)
    elif net_size == 101 : return resnet101(pretrained, channels_first)
    elif net_size == 152 : return resnet152(pretrained, channels_first)
    else:
        raise ValueError('Net size must be one of 18, 34, 50, 101, 152')
