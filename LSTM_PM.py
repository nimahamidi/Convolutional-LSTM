from keras.models import Model, Sequential
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, TimeDistributed, RepeatVector, LSTM, multiply, add
from keras.layers.convolutional import Conv2D
from keras.layers import UpSampling2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
import numpy as np
import time
import re
from keras.utils import plot_model


def relu(x):
    x = Activation('relu')(x)
    return x
def tanh(x):
    x = Activation('tanh')(x)
    return x
def sigmoid(x):
    x = Activation('sigmoid')(x)
    return x
def conv(x, nf, ks, name, weight_decay=0):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    x = Conv2D(nf, (ks, ks), padding='same', name=name,
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
        kernel_initializer=random_normal(stddev=0.01),
        bias_initializer=constant(0.0))(x)
    return x
def conv_b_t(x, nf, ks, name):
    x = Conv2D(nf, (ks, ks), padding='same', name=name, use_bias=True)(x)
    return x
def conv_b_f(x, nf, ks, name):
    x = Conv2D(nf, (ks, ks), padding='same', name=name, use_bias=False)(x)
    return x
def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st,st), name=name)(x)
    return x
def pool_center_lower (x, ks, st):
    x = AveragePooling2D((ks, ks), strides=(st,st))(x)
    return x

def convnet1(image, num_class):
    '''
    param image: 3 * 368 * 368
    return: initial_heatmap out_class * 45 * 45
    '''
    x = conv(image, 128, 9, 'conv1convnet1')
    x = relu(x)
    x = pooling(x, 3, 2, 'pool1_convnet1')   # output 128 * 184 * 184
    x = conv(x, 128, 9, 'conv2convnet1')
    x = relu(x)
    x = pooling(x, 3, 2, 'pool2_convnet1')   # output 128 * 92 * 92
    x = conv(x, 128, 9, 'conv3convnet1')
    x = relu(x)
    x = pooling(x, 3, 2, 'pool3_convnet1')   # output 128 * 45 * 45
    x = conv(x, 32, 5, 'conv4convnet1')
    x = relu(x) # output 32 * 45 * 45
    x = conv(x, 512, 9, 'conv5convnet1')
    x = relu(x) # output 512 * 45 * 45
    x = conv(x, 512, 1, 'conv6convnet1')
    x = relu(x) # output 512 * 45 * 45
    initial_heatmap = conv(x, num_class, 1, 'conv7convnet1')
    return initial_heatmap

def convnet2(image, num_class, s_num):
    '''
    param image: 3 * 368 * 368
    return: features 32 * 45 * 45
    '''
    x = conv(image, 128, 9, 'conv1convnet2_s%d' % s_num)
    x = relu(x)
    x = pooling(x, 3, 2, 'pool1_convnet2_s%d' % s_num)   # output 128 * 184 * 184
    x = conv(x, 128, 9, 'conv2convnet1_s%d' % s_num)
    x = relu(x)
    x = pooling(x, 3, 2, 'pool2_convnet2_s%d' % s_num)   # output 128 * 92 * 92
    x = conv(x, 128, 9, 'conv3convnet2_s%d' % s_num)
    x = relu(x)
    x = pooling(x, 3, 2, 'pool3_convnet2_s%d' % s_num)   # output 128 * 45 * 45

    x = conv(x, 32, 5, 'conv4convnet2_s%d' % s_num)
    x = relu(x) # output 32 * 45 * 45
    return x

def convnet3(hide_t, num_class, s_num):
    '''
    param h_t: 48 * 45 * 45
    return: heatmap  num_class * 45 * 45
    '''
    x = conv(hide_t, 128, 11, 'Mconv1convnet3_s%d' % s_num)
    x = relu(x)  # output 128 * 45 * 45
    x = conv(x, 128, 11, 'Mconv2convnet3_s%d' % s_num)
    x = relu(x)  # output 128 * 45 * 45
    x = conv(x, 128, 11, 'Mconv3convnet3_s%d' % s_num)
    x = relu(x) # output 128 * 45 * 45
    x = conv(x, 128, 1, 'Mconv4convnet3_s%d' % s_num)
    x = relu(x) # output 128 * 45 * 45

    x = conv(x, num_class, 1, 'Mconv5convnet3_s%d' % s_num)
    return x

def lstm(heatmap, features, centermap, hide_t_1, cell_t_1, s_num):
    '''
    param heatmap:     (class+1) * 45 * 45
    param features:    32 * 45 * 45
    param centermap:   1 * 45 * 45
    param hide_t_1:    48 * 45 * 45
    param cell_t_1:    48 * 45 * 45
    return:
        ide_t:    48 * 45 * 45
        cell_t:    48 * 45 * 45
    '''
    xt = Concatenate()([heatmap, features, centermap]) # (32+ class+1 +1 ) * 45 * 45

    gx = conv_b_t(xt, 48, 3, 'conv_gx_lstm_s%d' % s_num) #conv_b_t: bias=True - output: 48 * 45 * 45
    gh = conv_b_f(gx, 48, 3, 'conv_gh_lstm_s%d' % s_num) #conv_b_f: bias=False - output: 48 * 45 * 45
    g_sum = add([gx, gh])
    gt = tanh(g_sum)

    ox = conv_b_t(xt, 48, 3, 'conv_ox_lstm_s%d' % s_num) # output: 48 * 45 * 45
    oh = conv_b_f(ox, 48, 3, 'conv_oh_lstm_s%d' % s_num) # output: 48 * 45 * 45
    o_sum = add([ox, oh])
    ot = sigmoid(o_sum)

    ix = conv_b_t(xt, 48, 3, 'conv_ix_lstm_s%d' % s_num) # output: 48 * 45 * 45
    ih = conv_b_f(ix, 48, 3, 'conv_ih_lstm_s%d' % s_num) # output: 48 * 45 * 45
    i_sum = add([ix, ih])
    it = sigmoid(i_sum)

    fx = conv_b_t(xt, 48, 3, 'conv_fx_lstm_s%d' % s_num) # output: 48 * 45 * 45
    fh = conv_b_f(fx, 48, 3, 'conv_fh_lstm_s%d' % s_num) # output: 48 * 45 * 45
    f_sum = add([fx, fh])
    ft = sigmoid(f_sum)

    cell_t = add([multiply([ft, cell_t_1]), multiply([it,gt])])
    hide_t = multiply([ot, tanh(cell_t)])

    return cell_t, hide_t

#initial lstm
def lstm0(x):
    gx = conv(x, 48, 3, 'conv_gx_lstm0')
    gx = tanh(gx)
    ix = conv(x, 48, 3, 'conv_ix_lstm0')
    ix = sigmoid(ix)
    ox = conv(x, 48, 3, 'conv_ox_lstm0')
    ox = sigmoid(ox)

    cell_1 = tanh(add([gx, ix]))
    hide_1 = multiply([ox, cell_1])

    return cell_1, hide_1

def stageT(image, cmap, heatmap, cell_t_1, hide_t_1, num_class, s_num):
    '''
    param image:               3 * 368 * 368
    param cmap: gaussian       1 * 368 * 368
    param heatmap:             num_class * 45 * 45
    param cell_t_1:            48 * 45 * 45
    param hide_t_1:            48 * 45 * 45
    return:
        new_heatmap:                num_class * 45 * 45
        cell_t:                     48 * 45 * 45
        hide_t:                     48 * 45 * 45
    '''
    features = convnet2(image, num_class, s_num)
    centermap = pool_center_lower(cmap, 9, 8)
    cell_t, hide_t = lstm (heatmap, features, centermap, hide_t_1, cell_t_1, s_num)
    new_heat_map = convnet3(hide_t, num_class, s_num)
    return new_heat_map, cell_t, hide_t

def stage1(image, cmap, num_class):
    '''
    param image:                3 * 368 * 368
    param cmap:                 1 * 368 * 368
    return:
    heatmap:                     out_class * 45 * 45
    cell_t:                      48 * 45 * 45
    hide_t:                      48 * 45 * 45
    '''
    initial_heatmap = convnet1(image, num_class)
    features = convnet2(image, num_class, 1)
    centermap = pool_center_lower(cmap, 9, 8)

    x = Concatenate()([initial_heatmap, features, centermap])
    cell_1, hide_1 = lstm0(x)
    heatmap = convnet3(hide_1, num_class, 1)
    return initial_heatmap, heatmap, cell_1, hide_1

def forward(images, center_map, num_class, temporal):
    '''
    param images:      Tensor      (T * 3) * w(368) * h(368)
    param center_map:  Tensor      1 * 368 * 368
    return:
        heatmaps list (T + 1)* out_class * 45 * 45  includes the initial heatmap
    '''
    image = images[0]

    heat_maps = []
    initial_heatmap, heatmap, cell, hide = stage1(image, center_map, num_class)

    heat_maps.append(initial_heatmap)  # for initial loss
    heat_maps.append(heatmap)

    for i in range (2, temporal):
        image = images[i]
        heatmap, cell, hide = stageT(image, center_map, heatmap, cell, hide, num_class, i)
        print (image, heatmap)
        heat_maps.append(heatmap)

    return heat_maps

def get_test_model (num_class, temporal):
    input_images = []
    for i in range (temporal):
        instance = Input(shape=(None, None, 3))
        input_images.append(instance)
    center_map = Input(shape=(None, None, 1))
    input_images.append(center_map)
    output = forward (input_images, center_map, num_class, temporal)
    print ("out", output)
    model = Model(inputs=input_images, outputs=output)
    return model

def get_train_model (num_class, temporal):
    return
