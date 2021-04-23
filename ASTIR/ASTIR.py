import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import add, Activation, BatchNormalization, concatenate, Dropout, Dense, Conv2D, Reshape, GlobalAveragePooling2D, Permute, ConvLSTM2D


def squeeze_excite_block(input, len_seq, channel, map_height, map_width, data_format='channels_first', ratio=2, layer_count=0):
    with tf.name_scope('Squeeze_Block_{}'.format(layer_count)):
        init = Reshape((len_seq*channel, map_height, map_width))(input)
        channel_axis = 1 if data_format == "channels_first" else -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D(data_format=data_format)(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if data_format == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = tf.keras.layers.multiply([init, se])
        x = Reshape((len_seq, channel, map_height, map_width))(x)
    return x

def convLSTM_block(inputs, filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', data_format='channels_first', activation='tanh', dropout=0.0, l1_rec=0, l2_rec=0, l1_ker=0, l2_ker=0, l1_act=0, l2_act=0, return_sequences=True, use_bn=False):
    x = ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   data_format=data_format, return_sequences=return_sequences,
                                   recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=l1_rec, l2=l2_rec),
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_ker, l2=l2_ker),
                                   activity_regularizer=tf.keras.regularizers.l1_l2(l1=l1_act, l2=l2_act))(inputs)
    if use_bn:
        if data_format == 'channels_first':
            if return_sequences:
                channel_axes = 2
            else:
                channel_axes = 1
        else:
            channel_axes = -1
        x = BatchNormalization(axis=channel_axes, scale=False)(x)
    x = Activation(activation=activation)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    return x

def convLSTM_Inception_ResNet_module(inputs, layer_count, filters=32, strides=(1, 1), padding='same', data_format='channels_first', activation='tanh', types=1, dropout=0.0, l1_rec=0, l2_rec=0, l1_ker=0, l2_ker=0, l1_act=0, l2_act=0, use_bn=False, use_add_bn=True):
    with tf.name_scope('Inception_ResNet_ConvLSTM_Block_{}'.format(layer_count)):
        
        a = convLSTM_block(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

        a = convLSTM_block(a, filters=filters, kernel_size=(3, 3), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

        a = convLSTM_block(a, filters=filters, kernel_size=(3, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

        b = convLSTM_block(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

        b = convLSTM_block(b, filters=filters, kernel_size=(1, 5), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

        c = convLSTM_block(inputs, filters=filters, kernel_size=(5, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

        d = concatenate([a, b, c], axis=2)
        x = convLSTM_block(d, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

        x = tf.keras.layers.add([x, inputs])
        if use_add_bn:
            if data_format == 'channels_first':
                channel_axes = 2
            else:
                channel_axes = -1
            x = tf.keras.layers.BatchNormalization(axis=channel_axes, scale=False)(x)
    return x



def convLSTM_Inception_ResNet_network(
    c_conf=(3, 1, 32, 32), p_conf=(3, 1, 32, 32), t_conf=(3, 1, 32, 32), output_shape=(1, 32, 32), external_shape=(23,),
    nb_modules=2,
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format='channels_first',
    activation='tanh',
    dropout=0.2,
    dropout_inception_block=0,
    l1_rec=0, l2_rec=0, l1_ker=0, l2_ker=0, l1_act=0, l2_act=0, use_bn=False, use_add_bn=True,
    types=0
    ):
    inputs, outputs = [], []
    for input_shape in [c_conf, p_conf, t_conf]:
        if input_shape[0] > 0:
            len_seq, channel, map_height, map_width = input_shape
            input_img = Input(shape=(len_seq, channel, map_height, map_width))
            inputs.append(input_img)
            
            # ConvLSTM
            x = convLSTM_block(input_img, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=False)
            
            # ASTIR Block
            for i in range(nb_modules):
                x = convLSTM_Inception_ResNet_module(x, layer_count=i, filters=filters, strides=strides, padding=padding, dropout=dropout_inception_block,
                                                    data_format=data_format, activation=activation, types=types, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn, use_add_bn=True)
                x = squeeze_excite_block(input=x, len_seq=len_seq, channel=filters, map_height=map_height, map_width=map_width, data_format=data_format, ratio=2, layer_count=i)

            # ConvLSTM    
            x = convLSTM_block(x, filters=channel, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, return_sequences=False, use_bn=False)
        outputs.append(x)

    added = add(outputs)

    # external
    if len(external_shape) != None and len(external_shape) > 0:
        external = tf.keras.layers.Input(shape=external_shape)
        inputs.append(external)
        y = Dense(10, activation=activation)(external)
        y = Dense(output_shape[0] * output_shape[1] * output_shape[2], activation=activation)(y)
        y = Reshape(output_shape)(y)
        added = add([added, y])
    if use_add_bn:
        if data_format == 'channels_first':
            channel_axes = 1
        else:
            channel_axes = -1
        added = BatchNormalization(axis=channel_axes, scale=False)(added)
    result = Activation(activation='tanh')(added)
    model = Model(inputs=inputs, outputs=result)

    print(model.summary())

    return model

if __name__ == '__main__':
    # for types in range(3):
    model = convLSTM_Inception_ResNet_network()
    tf.keras.utils.plot_model(model, to_file="./ASTIR.png", show_shapes=True)

