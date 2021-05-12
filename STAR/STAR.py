from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import add, Dense, Conv2D, Reshape, Concatenate, Activation, BatchNormalization


def _shortcut(input, residual):
    return add([input, residual])

# 2D
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:                                                                                                                                                                                                                                                                                                                                                                                                              
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)

        return Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, padding="same")(activation)                                                                                                                                                                                                                                        
    return f

def ResUnits2D(residual_unit, nb_filter, map_height=16, map_width=8, repetations=1):
    def f(input):
        for i in range(repetations): 
            init_subsample = (1, 1)
            input = _residual_unit(nb_filter=nb_filter, init_subsample=init_subsample)(input)

        return input
    return f

def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f

def STAR(c_conf=(3, 2, 32, 32), p_conf=(1, 2, 32, 32), t_conf=(1, 2, 32, 32), external_dim=8, nb_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''
    map_height, map_width = 32, 32
    nb_flow = 2
    nb_filter = 64

    main_inputs = []

    input = tf.keras.Input(shape=(map_height, map_width, (nb_flow * (c_conf[0]+p_conf[0]+t_conf[0]))))

    main_inputs.append(input)
    main_output = input # flows data

    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10, activation='relu')(external_input)
        h1 = Dense(units=2*map_height * map_width, activation='relu')(embedding)
        external_output = Reshape((map_height, map_width, 2))(h1)
        main_output = Concatenate(axis=-1)([input, external_output])
    else:
        print('external_dim:', external_dim)

    # step 1:Conv
    conv1 = Conv2D(nb_filter, (3, 3), padding="same")(main_output)

    # step 2: n * RB(residual block) 'relu -> conv -> relu ->conv'
    residual_output = ResUnits2D(_residual_unit, nb_filter=nb_filter, repetations=nb_residual_unit)(conv1)
    activation = Activation('relu')(residual_output)

    # step 3: Conv
    conv2 = Conv2D(nb_flow, (3, 3), padding='same')(activation)

    # step 4: Tanh
    main_output = Activation('tanh')(conv2)

    model = Model(inputs=main_inputs, outputs=main_output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == '__main__':
    model = STAR(external_dim=8, nb_residual_unit=2)
    tf.keras.utils.plot_model(model, to_file="./STAR.png", show_shapes=True)
    model.summary()
