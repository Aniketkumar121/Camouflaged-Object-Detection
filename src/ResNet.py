from tensorflow.keras.layers import (Conv2D,
                                        BatchNormalization,
                                        Activation,
                                        add)


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters
    
    bn_axis = -1
    
    conv_name_base = 'conv' + str(stage) + '_' + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name='conv' + str(stage) + block + '1_conv')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='conv' + str(stage) + block + '1_bn')(x)
    x = Activation('relu', name='conv' + str(stage) + block + '1_relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name='conv' + str(stage) + block + '2_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='conv' + str(stage) + block + '2_bn')(x)
    x = Activation('relu', name='conv' + str(stage) + block + '2_relu')(x)

    x = Conv2D(filters3, (1, 1), name='conv' + str(stage) + block + '3_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='conv' + str(stage) + block + '3_bn')(x)

    x = add([x, input_tensor], name='conv' + str(stage) + block + 'add')
    x = Activation('relu', name='conv' + str(stage) + block + 'out')(x)
    return x



def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    bn_axis = -1
    
    conv_name_base = 'conv' + str(stage) + block +'conv'
    bn_name_base = 'conv' + str(stage) + block + 'nb'

    x = Conv2D(filters1, (1, 1), strides=strides, name='conv' + str(stage) + block + '1_conv')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='conv' + str(stage) + block + '1_bn')(x)
    x = Activation('relu', name='conv' + str(stage) + block + '1_relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name='conv' + str(stage) + block + '2_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='conv' + str(stage) + block + '2_bn')(x)
    x = Activation('relu', name='conv' + str(stage) + block + '2_relu')(x)

    x = Conv2D(filters3, (1, 1), name='conv' + str(stage) + block + '0_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='conv' + str(stage) + block + '0_bn')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name='conv' + str(stage) + block + '3_conv')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name='conv' + str(stage) + block + '3_bn')(shortcut)

    x = add([x, shortcut], name='conv' + str(stage) + block + 'add')
    x = Activation('relu', name='conv' + str(stage) + block + 'out')(x)
    return x

def layer1(x):
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='_block1_', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='_block2_')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='_block3_')
    return x


def layer2(x):
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='_block1_')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='_block2_')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='_block3_')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='_block4_')
    return x

def layer3_1(x):
    x1 = conv_block(x, 3, [256, 256, 1024], stage=4, block='_block1_')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='_block2_')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='_block3_')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='_block4_')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='_block5_')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='_block6_')
    return x1

def layer4_1(x):
    x1 = conv_block(x, 3, [512, 512, 2048], stage=5, block='_block1_')
    x1 = identity_block(x1, 3, [512, 512, 2048], stage=5, block='_block2_')
    x1 = identity_block(x1, 3, [512, 512, 2048], stage=5, block='_block3_')
    return x1

def layer3_2(x):
    x2 = conv_block(x, 3, [256, 256, 1024], stage=4, block='_block21_')
    x2 = identity_block(x2, 3, [256, 256, 1024], stage=4, block='_block22_')
    x2 = identity_block(x2, 3, [256, 256, 1024], stage=4, block='_block23_')
    x2 = identity_block(x2, 3, [256, 256, 1024], stage=4, block='_block24_')
    x2 = identity_block(x2, 3, [256, 256, 1024], stage=4, block='_block25_')
    x2 = identity_block(x2, 3, [256, 256, 1024], stage=4, block='_block26_')
    return x2

def layer4_2(x):
    x2 = conv_block(x, 3, [512, 512, 2048], stage=5, block='_block21_')
    x2 = identity_block(x2, 3, [512, 512, 2048], stage=5, block='_block22_')
    x2 = identity_block(x2, 3, [512, 512, 2048], stage=5, block='_block23_')
    return x2


