import tensorflow as tf
import tensorflow.keras.layers as layers

from utils.rearrange import resnet_weight

tf.random.set_seed(0)
L2_WEIGHT_DECAY=1e-4
BATCH_NORM_DECAY = 0.99
BATCH_NORM_EPSILON = 1e-3

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

        self.conv1 = layers.Conv2D(channel, kernel_size=3, strides=stride, padding='same', use_bias=False, data_format=data_format, kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization(axis=axis, gamma_initializer='ones', beta_initializer='zeros')
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(channel, kernel_size=3, padding='same', use_bias=False, data_format=data_format, kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization(axis=axis, gamma_initializer='ones', beta_initializer='zeros')
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

    def __init__(self, channel, stride=1, downsample=None, channels_first=True, weight_dict=None):
        super(Bottleneck, self).__init__()
        if channels_first :
            data_format = 'channels_first'
            axis=1
        else :
            data_format = 'channels_last'
            axis=3

        if weight_dict is not None:
            self.stride = stride
            self.conv1 = layers.Conv2D(channel, kernel_size=1, padding='same', use_bias=False, data_format=data_format, kernel_initializer='he_normal', weights=[weight_dict['conv1']['weights']])
            self.bn1 = layers.BatchNormalization(axis=axis,
                                                momentum=BATCH_NORM_DECAY,
                                                epsilon=BATCH_NORM_EPSILON
                                )
            self.relu1 = layers.ReLU()

            self.conv2 = layers.Conv2D(channel, kernel_size=3, strides=stride, padding='same', use_bias=False, data_format=data_format, kernel_initializer='he_normal', weights=[weight_dict['conv2']['weights']])
            self.bn2 = layers.BatchNormalization(axis=axis,
                                                momentum=BATCH_NORM_DECAY,
                                                epsilon=BATCH_NORM_EPSILON
                                )
            self.relu2 = layers.ReLU()

            self.conv3 = layers.Conv2D(channel*4, kernel_size=1, padding='same', use_bias=False, data_format=data_format, kernel_initializer='he_normal', weights=[weight_dict['conv3']['weights']])
            self.bn3 = layers.BatchNormalization(axis=axis,
                                                momentum=BATCH_NORM_DECAY,
                                                epsilon=BATCH_NORM_EPSILON
                                )
            self.relu3 = layers.ReLU()

        else:
            self.stride = stride
            self.conv1 = layers.Conv2D(channel,
                                        kernel_size=1,
                                        padding='same',
                                        use_bias=False,
                                        data_format=data_format,
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                )
            self.bn1 = layers.BatchNormalization(axis=axis,
                                                momentum=BATCH_NORM_DECAY,
                                                epsilon=BATCH_NORM_EPSILON
                                )

            self.conv2 = layers.Conv2D(channel,
                                        kernel_size=3,
                                        strides=stride,
                                        padding='same',
                                        use_bias=False,
                                        data_format=data_format,
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                )
            self.bn2 = layers.BatchNormalization(axis=axis,
                                                momentum=BATCH_NORM_DECAY,
                                                epsilon=BATCH_NORM_EPSILON
                                )

            self.conv3 = layers.Conv2D(channel*4,
                                        kernel_size=1,
                                        padding='same',
                                        use_bias=False,
                                        data_format=data_format,
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                )
            self.bn3 = layers.BatchNormalization(axis=axis,
                                                momentum=BATCH_NORM_DECAY,
                                                epsilon=BATCH_NORM_EPSILON
                                )

        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = tf.nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += residual
        out = tf.nn.relu(out)

        return out

class ResNet(tf.keras.Model):
    def __init__(self, block, layer, channels_first=True, weight_dict=None):
        self.inplanes = 64

        super(ResNet, self).__init__()
        if channels_first :
            self.data_format = 'channels_first'
            self.axis=1
        else :
            self.data_format = 'channels_last'
            self.axis=3

        if weight_dict is not None:
            self.training=False
            self.conv1 = layers.Conv2D(64, kernel_size=7,
                                    strides=2,
                                    padding='same',
                                    use_bias=False,
                                    data_format=self.data_format,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                    weights=[weight_dict['conv1']['weights']]
                            )
            self.bn1 = layers.BatchNormalization(axis=self.axis,
                                                 momentum=BATCH_NORM_DECAY,
                                                 epsilon=BATCH_NORM_EPSILON
                            )
            self.maxpool = layers.MaxPool2D(pool_size=3,
                                            strides=2,
                                            padding='same',
                                            data_format=self.data_format
                            )

            self.layer1 = self._make_layer(block, 64, layer[0], weight_dict=weight_dict['block1'])
            self.layer2 = self._make_layer(block, 128, layer[1], stride=2, weight_dict=weight_dict['block2'])
            self.layer3 = self._make_layer(block, 256, layer[2], stride=2, weight_dict=weight_dict['block3'])
            self.layer4 = self._make_layer(block, 512, layer[3], stride=2, weight_dict=weight_dict['block4'])

        else:
            self.training=True
            self.conv1 = layers.Conv2D(64, kernel_size=7,
                                    strides=2,
                                    padding='same',
                                    use_bias=False,
                                    data_format=self.data_format,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY)
                            )
            self.bn1 = layers.BatchNormalization(axis=self.axis,
                                                 momentum=BATCH_NORM_DECAY,
                                                 epsilon=BATCH_NORM_EPSILON
                            )
            self.maxpool = layers.MaxPool2D(pool_size=3,
                                            strides=2,
                                            padding='same',
                                            data_format=self.data_format
                            )

            self.layer1 = self._make_layer(block, 64, layer[0])
            self.layer2 = self._make_layer(block, 128, layer[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layer[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layer[3], stride=2)

        # self.avg_pool = layers.AveragePooling2D(pool_size, strides, padding, data_format=self.data_format)
        self.avg_pool = layers.GlobalAveragePooling2D(data_format=self.data_format)
        # self.reshape = layers.Reshape((512 * block.expansion, ))
        # self.fc = layers.Dense(10, use_bias=False)
        self.fc = layers.Dense(10,
                                activation='softmax',
                                use_bias=True,
                                kernel_initializer='he_normal',
                                # bias_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                    )


        self.size =[64,
                    64 * block.expansion,
                    128 * block.expansion,
                    256 * block.expansion,
                    512 * block.expansion]

    def _make_layer(self, block, channel, blocks, stride=1, weight_dict=None):
        downsample = None

        if stride != 1 or self.inplanes != channel * block.expansion:
            downsample = tf.keras.Sequential()
            downsample.add(layers.Conv2D(
                                        channel * block.expansion,
                                        kernel_size=1,
                                        strides=stride,
                                        use_bias=False,
                                        data_format=self.data_format,
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                        ))
            downsample.add(layers.BatchNormalization(
                                        axis=self.axis,
                                        momentum=BATCH_NORM_DECAY,
                                        epsilon=BATCH_NORM_EPSILON
                        ))

        layer = tf.keras.Sequential()
        if weight_dict is not None:
            layer.add(block(channel, stride, downsample, weight_dict=weight_dict[1]))
            for i in range(1, blocks):
                layer.add(block(channel, weight_dict=weight_dict[i+1]))

        else:
            layer.add(block(channel, stride, downsample))
            for i in range(1, blocks):
                layer.add(block(channel))

        return layer

    def call(self, input):
        # C1 = self.maxpool(tf.nn.relu(self.bn1(self.conv1(input), training)))
        C1 = self.maxpool(tf.nn.relu(self.bn1(self.conv1(input))))

        C2 = self.layer1(C1)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        # out = self.reshape(C5)
        out = self.avg_pool(C5)
        out = self.fc(out)
        # out = self.fc2(out)
        # print(out)

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
    if pretrained:
        import yaml
        pretrain_dir = yaml.load(open('./configs/pretrain.yaml'))
        pretrain_dir = pretrain_dir['resnet50']
        weight_dict = resnet_weight(pretrain_dir)
        model = ResNet(Bottleneck, [3, 4, 6, 3], channels_first=channels_first, weight_dict=weight_dict)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], channels_first=channels_first)
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
