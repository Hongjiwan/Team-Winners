import torch
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dense, BatchNormalization
from keras.layers import Add, Reshape, Multiply
import keras.backend as K

def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    x = Conv2D(filters, kernel_size, kernel_initializer='he_normal', padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    
    return x


def SE_block(input_tensor, reduction_ratio=16):
    ch_input = K.int_shape(input_tensor)[-1]
    ch_reduced = ch_input//reduction_ratio
    
    # Squeeze
    x = GlobalAveragePooling2D()(input_tensor) # Eqn.2
    
    # Excitation
    x = Dense(ch_reduced, kernel_initializer='he_normal', activation='relu', use_bias=False)(x) # Eqn.3
    x = Dense(ch_input, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)(x) # Eqn.3
    
    x = Reshape( (1, 1, ch_input) )(x)
    x = Multiply()([input_tensor, x]) # Eqn.4
    
    return x
   

def SE_residual_block(**kwargs, input_tensor, filter_sizes, strides=1, reduction_ratio=16):
    filter_1, filter_2, filter_3 = filter_sizes
    
    x = conv2d_bn(input_tensor, filter_1, (1, 1), strides=strides)
    x = conv2d_bn(x, filter_2, (3, 3))
    x = conv2d_bn(x, filter_3, (1, 1), activation=None)
    
    x = SE_block(x, reduction_ratio)
    
    projected_input = conv2d_bn(input_tensor, filter_3, (1, 1), strides=strides, activation=None) if K.int_shape(input_tensor)[-1] != filter_3 else input_tensor
    shortcut = Add()([projected_input, x])
    shortcut = Activation(activation='relu')(shortcut)
    
    return shortcut
 

def stage_block(input_tensor, filter_sizes, blocks, reduction_ratio=16, stage=''):
    strides = 2 if stage != '2' else 1
    
    x = SE_residual_block(input_tensor, filter_sizes, strides, reduction_ratio) # projection layer

    for i in range(blocks-1):
        x = SE_residual_block(x, filter_sizes, reduction_ratio=reduction_ratio)
    
    return x
    

def SE_ResNeXt_WSL(model_input, classes=10):
    stage_1 = conv2d_bn(model_input, 64, (7, 7), strides=2, padding='same') # (112, 112, 64)
    stage_1 = MaxPooling2D((3, 3), strides=2, padding='same')(stage_1) # (56, 56, 64)
    
    stage_2 = stage_block(stage_1, [512, 512, 256], 3, reduction_ratio=16, stage='2')
    stage_3 = stage_block(stage_2, [1024, 1024, 512], 4, reduction_ratio=16, stage='3') # (28, 28, 512)
    stage_4 = stage_block(stage_3, [2048, 2048, 1024], 23, reduction_ratio=16, stage='4') # (14, 14, 1024)
    stage_5 = stage_block(stage_4, [4096, 4096, 2048], 3, reduction_ratio=16, stage='5') # (7, 7, 2048)

    gap = GlobalAveragePooling2D()(stage_5)
    
    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(gap) # 'softmax'
    
    model = Model(inputs=model_input, outputs=model_output, name='SE-ResNeXt-WSL')
        
    return model
