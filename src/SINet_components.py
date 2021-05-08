from tensorflow.keras import activations, Input, Model
from tensorflow.keras.layers import (Conv2D,
                                    BatchNormalization,
                                    Activation,
                                    Concatenate,
                                    UpSampling2D,
                                    ZeroPadding2D,)

# Basic Convolution ( Conv + Batch Normalization + ReLu ) in Functional API
def basic_conv2d_2(input_layer, filters, kernel_size, strides=1, padding=0, dilation_rate=1):
    pad1 = ZeroPadding2D(padding=padding)(input_layer)
    conv1 = Conv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        dilation_rate=dilation_rate)(pad1)
    batch1 = BatchNormalization()(conv1)
    output_layer = Activation(activations.relu)(batch1)
    return output_layer


# Receptive Field in Functional API
def RF(input_shape, filters):
    
    input_layer = Input(shape=input_shape)
    # Block 0
    branch0 = basic_conv2d_2(input_layer=input_layer, filters=filters, kernel_size=1)
    
    # Block 1
    branch1 = basic_conv2d_2(input_layer=input_layer, filters=filters, kernel_size=1)
    branch1 = basic_conv2d_2(input_layer=branch1, filters=filters, kernel_size=(1, 3), padding=(0, 1))
    branch1 = basic_conv2d_2(input_layer=branch1, filters=filters, kernel_size=(3, 1), padding=(1, 0))
    branch1 = basic_conv2d_2(input_layer=branch1, filters=filters, kernel_size=3, padding=3, dilation_rate=3)

    # Block 2
    branch2 = basic_conv2d_2(input_layer=input_layer, filters=filters, kernel_size=1)
    branch2 = basic_conv2d_2(input_layer=branch2, filters=filters, kernel_size=(1, 5), padding=(0, 2))
    branch2 = basic_conv2d_2(input_layer=branch2, filters=filters, kernel_size=(5, 1), padding=(2, 0))
    branch2 = basic_conv2d_2(input_layer=branch2, filters=filters, kernel_size=3, padding=5, dilation_rate=5)
    
    # Block 3
    branch3 = basic_conv2d_2(input_layer=input_layer, filters=filters, kernel_size=1)
    branch3 = basic_conv2d_2(input_layer=branch3, filters=filters, kernel_size=(1, 7), padding=(0, 3))
    branch3 = basic_conv2d_2(input_layer=branch3, filters=filters, kernel_size=(7, 1), padding=(3, 0))
    branch3 = basic_conv2d_2(input_layer=branch3, filters=filters, kernel_size=3, padding=7, dilation_rate=7)
    
    # Concatenate models
    merge = Concatenate()([branch0, branch1, branch2, branch3])
    
    # Block 1 of Receptive Field
    conv_cat = basic_conv2d_2(input_layer=merge, filters=filters, kernel_size=3, padding=1)
    
    # Block 2 of Receptive Field:
    conv_res = basic_conv2d_2(input_layer=input_layer, filters=32, kernel_size=1)

    # Final addition
    rf = Activation(activations.relu)(conv_cat+conv_res)

    # Build final Receptive Field model:
    RF = Model(inputs=input_layer, outputs=rf)
    
    return RF


# Search Module
def PDC_SM(shape1, shape2, shape3, shape4, channel=32):
    
    x1 = Input(shape=shape1)
    x2 = Input(shape=shape2)
    x3 = Input(shape=shape3)
    x4 = Input(shape=shape4)

    # Layers
    up_sampling = UpSampling2D(size=2, interpolation='bilinear')
    

    conv_upsample1 = lambda x: basic_conv2d_2(input_layer=x, filters=channel, kernel_size=3, padding=1)
    conv_upsample2 = lambda x: basic_conv2d_2(input_layer=x, filters=channel, kernel_size=3, padding=1)
    conv_upsample3 = lambda x: basic_conv2d_2(input_layer=x, filters=channel, kernel_size=3, padding=1)
    conv_upsample4 = lambda x: basic_conv2d_2(input_layer=x, filters=channel, kernel_size=3, padding=1)
    conv_upsample5 = lambda x: basic_conv2d_2(input_layer=x, filters=2*channel, kernel_size=3, padding=1)
    
    conv_concat1 = lambda x: basic_conv2d_2(input_layer=x, filters=2*channel, kernel_size=3, padding=1)
    conv_concat2 = lambda x: basic_conv2d_2(input_layer=x, filters=4*channel, kernel_size=3, padding=1)

    conv4 = lambda x: basic_conv2d_2(input_layer=x, filters=4*channel, kernel_size=3, padding=1)
    conv5 = Conv2D(filters=1, kernel_size=1)
    
    # Model 
    x1_1 = x1
    
    x2_1 = conv_upsample1(up_sampling(x1)) * x2
    
    x3_1 = conv_upsample2(up_sampling(up_sampling(x1))) * conv_upsample3(up_sampling(x2)) * x3
    
    x2_2 = Concatenate(axis=-1)([x2_1, conv_upsample4(up_sampling(x1_1))])
    x2_2 = conv_concat1(x2_2)

    x3_2 = Concatenate(axis=-1)([x3_1,  conv_upsample5(up_sampling(x2_2)), x4])
    x3_2 = conv_concat2(x3_2)

    x = conv4(x3_2)
    x = conv5(x)
    
    model = Model(inputs=[x1, x2, x3, x4], outputs=x)
    
    return model


# Identification module
def PDC_IM(shape1, shape2, shape3, channel):
    
    x1 = Input(shape=shape1)
    x2 = Input(shape=shape2)
    x3 = Input(shape=shape3)
    
    # Layers
    up_sampling = UpSampling2D(size=2, interpolation='bilinear')
    

    conv_upsample1 = lambda x: basic_conv2d_2(input_layer=x, filters=channel, kernel_size=3, padding=1)
    conv_upsample2 = lambda x: basic_conv2d_2(input_layer=x, filters=channel, kernel_size=3, padding=1)
    conv_upsample3 = lambda x: basic_conv2d_2(input_layer=x, filters=channel, kernel_size=3, padding=1)
    conv_upsample4 = lambda x: basic_conv2d_2(input_layer=x, filters=channel, kernel_size=3, padding=1)
    conv_upsample5 = lambda x: basic_conv2d_2(input_layer=x, filters=2*channel, kernel_size=3, padding=1)
    
    conv_concat2 = lambda x: basic_conv2d_2(input_layer=x, filters=2*channel, kernel_size=3, padding=1)
    conv_concat3 = lambda x: basic_conv2d_2(input_layer=x, filters=3*channel, kernel_size=3, padding=1)

    conv4 = lambda x: basic_conv2d_2(input_layer=x, filters=3*channel, kernel_size=3, padding=1)
    conv5 = Conv2D(filters=1, kernel_size=1)
    
    # Model 
    # X1_1
    x1_1 = x1

    # X2_1
    x2_1 = conv_upsample1(up_sampling(x1)) * x2

    # X3_1
    x3_1 = conv_upsample2(up_sampling(up_sampling(x1))) * conv_upsample3(up_sampling(x2)) * x3

    # X2_2
    x2_2 = Concatenate(axis=-1)([x2_1, conv_upsample4(up_sampling(x1_1))])
    x2_2 = conv_concat2(x2_2)

    # X3_2
    x3_2 = Concatenate(axis=-1)([x3_1,  conv_upsample5(up_sampling(x2_2))])
    x3_2 = conv_concat3(x3_2)

    x = conv4(x3_2)
    x = conv5(x)
    
    model = Model(inputs=[x1, x2, x3], outputs=x)
    
    return model