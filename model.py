import numpy as np
import tensorflow.keras as keras
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def first_encoding_block(inp_layer,network_param):
    SKIP_CONNECTION = inp_layer
    if inp_layer.shape[-1]!=network_param['filters']:
        SKIP_CONNECTION = keras.layers.Conv2D(filters     = network_param['filters'],
                                              kernel_size = (1,1),
                                              padding     ='same')(inp_layer)
    CONV1 = keras.layers.Conv2D(filters     = network_param['filters'],
                                kernel_size = network_param['kernel'],
                                strides     = network_param['strides'],
                                padding     ='same')(inp_layer)
    BN    = keras.layers.BatchNormalization()(CONV1)
    RELU  = keras.layers.ReLU()(BN)
    CONV2 = keras.layers.Conv2D(filters     = network_param['filters'],
                                kernel_size = network_param['kernel'],
                                strides     = network_param['strides'],
                                padding='same')(RELU)
    ADD   = keras.layers.Add()([SKIP_CONNECTION,CONV2])
    return ADD

def encoding_block(inp_layer,network_param):
    SKIP_CONNECTION = inp_layer
    if inp_layer.shape[-1]!=network_param['filters']:
        SKIP_CONNECTION = keras.layers.Conv2D(filters     = network_param['filters'],
                                              kernel_size = (1,1),
                                              strides     = network_param['skip_stride'],
                                              padding     ='same')(inp_layer)
    BN = keras.layers.BatchNormalization()(inp_layer)
    RELU  = keras.layers.ReLU()(BN)
    CONV1 = keras.layers.Conv2D(filters     = network_param['filters'],
                                kernel_size = network_param['kernel'],
                                strides     = network_param['strides']+1,
                                padding     ='same')(RELU)
    BN    = keras.layers.BatchNormalization()(CONV1)
    RELU  = keras.layers.ReLU()(BN)
    CONV2 = keras.layers.Conv2D(filters     = network_param['filters'],
                                kernel_size = network_param['kernel'],
                                strides     = network_param['strides'],
                                padding='same')(RELU)
    ADD   = keras.layers.Add()([SKIP_CONNECTION,CONV2])
    return ADD

def decoding_block(inp_layer,encoder_connection,network_param):
    UPSAMPLE    = keras.layers.UpSampling2D()(inp_layer)
    CONCATENATE = keras.layers.Concatenate()([UPSAMPLE,encoder_connection])
    SKIP_CONNECTION = CONCATENATE
    if inp_layer.shape[-1]!=network_param['filters']:
        SKIP_CONNECTION = keras.layers.Conv2D(filters     = network_param['filters'],
                                              kernel_size = (1,1),
                                              strides     = network_param['skip_stride'],
                                              padding     ='same')(CONCATENATE)
    BN = keras.layers.BatchNormalization()(CONCATENATE)
    RELU  = keras.layers.ReLU()(BN)
    CONV1 = keras.layers.Conv2D(filters     = network_param['filters'],
                                kernel_size = network_param['kernel'],
                                strides     = network_param['strides'],
                                padding     ='same')(RELU)
    BN    = keras.layers.BatchNormalization()(CONV1)
    RELU  = keras.layers.ReLU()(BN)
    CONV2 = keras.layers.Conv2D(filters     = network_param['filters'],
                                kernel_size = network_param['kernel'],
                                strides     = network_param['strides'],
                                padding='same')(RELU)
    ADD   = keras.layers.Add()([SKIP_CONNECTION,CONV2])
    return ADD

def deep_resnet():
    network_param = [
                        {'filters':64,'kernel':(3,3),'strides':1,'skip_stride':1},
                        {'filters':128,'kernel':(3,3),'strides':1,'skip_stride':2},
                        {'filters':256,'kernel':(3,3),'strides':1,'skip_stride':2},
                        {'filters':512,'kernel':(3,3),'strides':1,'skip_stride':2},
                        {'filters':256,'kernel':(3,3),'strides':1,'skip_stride':1},
                        {'filters':128,'kernel':(3,3),'strides':1,'skip_stride':1},
                        {'filters':64,'kernel':(3,3),'strides':1,'skip_stride':1}
                    ]
    INP_LAYER = keras.Input(shape=(224,224,3))
    ENCODER_BLOCK1 = first_encoding_block(INP_LAYER,network_param[0])
    ENCODER_BLOCK2 = encoding_block(ENCODER_BLOCK1,network_param[1])
    ENCODER_BLOCK3 = encoding_block(ENCODER_BLOCK2,network_param[2])
    BRIDGE = encoding_block(ENCODER_BLOCK3,network_param[3])
    DECODER_BLOCK1 = decoding_block(BRIDGE,ENCODER_BLOCK3,network_param[4])
    DECODER_BLOCK2 = decoding_block(DECODER_BLOCK1,ENCODER_BLOCK2,network_param[5])
    DECODER_BLOCK3 = decoding_block(DECODER_BLOCK2,ENCODER_BLOCK1,network_param[6])
    
    CONV_OUT = keras.layers.Conv2D(filters=1,kernel_size=(1,1),padding='same')(DECODER_BLOCK3)
    OUT      = keras.layers.Activation(keras.activations.sigmoid)(CONV_OUT)
    NETWORK = keras.Model(INP_LAYER,OUT)
    keras.utils.plot_model(NETWORK,to_file='deep_resnet.png',show_shapes=True)
    return NETWORK