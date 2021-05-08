from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Conv2D,
                                        BatchNormalization,
                                        Activation,
                                        Concatenate,
                                        UpSampling2D,
                                        ZeroPadding2D,
                                        MaxPooling2D)

from tensorflow.keras.activations import sigmoid

from ResNet import layer1, layer2, layer3_1, layer4_1, layer3_2, layer4_2
from Search_attention import SA
from SINet_components import RF, PDC_SM, PDC_IM

def SINet_ResNet50(image_size=352, channel=32, batch_size=12, opt=None):
    x = Input(shape=(image_size, image_size, 3))
    
    down_sample = lambda x: MaxPooling2D(2, strides=2)(x)
    upsample_2 = lambda x: UpSampling2D(size=2, interpolation='bilinear')(x)
    upsample_8 = lambda x: UpSampling2D(size=8, interpolation='bilinear')(x)

    # Head
    x0 = ZeroPadding2D(3, name='conv1_pad')(x)
    x0 = Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv')(x0)
    x0 = BatchNormalization(name='conv1_bn')(x0)
    
    # Low level features
    x0 = Activation('relu', name='conv1_relu')(x0) 
    x0 = ZeroPadding2D((1, 1), name='pool1_pad')(x0)
    x0 = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_pool')(x0)
    
    x1 = layer1(x0)
    x2 = layer2(x1)
    
    # Search Module (SM):
    
    x01 = Concatenate()([x0, x1])
    x01_down = down_sample(x01)

    rf_low_sm = RF(x01_down.shape[1:], channel)
    x01_sm_rf = rf_low_sm(x01_down)
    
    x2_sm = x2
    x3_sm = layer3_1(x2_sm)
    x4_sm = layer4_1(x3_sm)
    
    x2_sm_cat = Concatenate()([x2_sm, upsample_2(x3_sm), upsample_2(upsample_2(x4_sm))])    
    x3_sm_cat = Concatenate()([x3_sm, upsample_2(x4_sm)])
    
    rf2_sm = RF(x2_sm_cat.shape[1:], channel)
    x2_sm_rf = rf2_sm(x2_sm_cat)
    
    rf3_sm = RF(x3_sm_cat.shape[1:], channel)
    x3_sm_rf = rf3_sm(x3_sm_cat)
    
    rf4_sm = RF(x4_sm.shape[1:], channel)
    x4_sm_rf = rf4_sm(x4_sm)
    
    pdc_sm = PDC_SM(x4_sm_rf.shape[1:], x3_sm_rf.shape[1:], x2_sm_rf.shape[1:], x01_sm_rf.shape[1:], channel)
    camouflage_map_sm = pdc_sm([x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf])
    
    # Search Attention (SA)

    tmp = sigmoid(camouflage_map_sm)

    x2_sa = SA(tmp, x2)
    x2_sa = x2

    # Identification Module (IM)
    x3_im = layer3_2(x2_sa)
    x4_im = layer4_2(x3_im)

    rf2_im = RF(x2_sa.shape[1:], channel)
    x2_im_rf = rf2_im(x2_sa)

    rf3_im = RF(x3_im.shape[1:], channel)
    x3_im_rf = rf3_im(x3_im)

    rf4_im = RF(x4_im.shape[1:], channel)
    x4_im_rf = rf4_im(x4_im)

    pdc_im = PDC_IM(x4_im_rf.shape[1:], x3_im_rf.shape[1:], x2_im_rf.shape[1:], channel)
  
    camouflage_map_im = pdc_im([x4_im_rf, x3_im_rf, x2_im_rf])

    camouflage_map_sm = UpSampling2D(size=8, interpolation='bilinear', name='SM')(camouflage_map_sm)
    camouflage_map_im = UpSampling2D(size=8, interpolation='bilinear', name='IM')(camouflage_map_im)

    output_model = Model(x, outputs=[camouflage_map_sm, camouflage_map_im])
    return output_model