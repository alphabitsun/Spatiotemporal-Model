'''
    ST-ResNet: Deep Spatio-temporal Residual Networks
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf 
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import add, Activation, Dense, Conv2D, Reshape


def Hadamard(nb_flow, nb_row, nb_col):
    def f(input):
        w = np.random.random([nb_row, nb_col])
        # print(w.shape)
        w = tf.Variable(w, dtype=tf.float32)
        out = w * Reshape((nb_flow, nb_row, nb_col))(input)
        return Reshape((nb_row, nb_col, nb_flow))(out)
    return f


def _shortcut(input, residual):
    return add([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, padding="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter, init_subsample=init_subsample)(input)
        return input
    return f


def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''
    nb_flow, nb_row, nb_col = c_conf[1], c_conf[2], c_conf[3]
    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(map_height, map_width, nb_flow * len_seq))
            main_inputs.append(input)
            # Conv1
            conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(input)

            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Conv2D(filters=nb_flow, kernel_size=(3, 3), padding="same")(activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        # 直接合并
        # main_output = add([outputs[0],outputs[1],outputs[2]])
        # 带权值合并
        new_outputs = []
        for output in outputs:
            output = Hadamard(nb_flow, nb_row, nb_col)(output)
            new_outputs.append(output)
        main_output = add([new_outputs[0], new_outputs[1], new_outputs[2]])

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape(( map_height, map_width,nb_flow))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

if __name__ == '__main__':
    model = stresnet(external_dim=8, nb_residual_unit=2)
    tf.keras.utils.plot_model(model, to_file="./ST-ResNet.png", show_shapes=True)
    model.summary()
