import numpy as np
import tensorflow as tf 
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add, Activation, BatchNormalization, Conv2D, Dense, Concatenate, Reshape, Flatten, Multiply

def BasicBlock():
    def f(input):
        output = BatchNormalization()(input)
        output = Activation('relu')(output)
        output = Conv2D(filters=64, kernel_size=3, padding='same', data_format='channels_first')(output)
        output = BatchNormalization()(input)
        output = Activation('relu')(output)
        # print(x.shape[0])
        output = Conv2D(filters=64, kernel_size=3, padding='same', data_format='channels_first')(output)
        return Add()([input, output])
    return f


def FCN(nb_residual_unit=2):
    def f(input):
        y = Conv2D(filters=64, kernel_size=3, padding='same', data_format='channels_first')(input)
        for i in range(nb_residual_unit):
            y = BasicBlock()(y)
        output = Conv2D(filters=64, kernel_size=3, padding='same', data_format='channels_first')(y)
        return output
    return f

def MDL(node_conf=(3,2,8,8),node_tconf=(1,2,8,8),node_pconf=(1,2,8,8),
                 edge_conf=(3,512*3,8,8),edge_tconf=(1,512,8,8),edge_pconf=(1,512,8,8), external_dim=28):

    all_inputs = []
    all_outputs1 = []
    all_outputs2 = []
    X_node = [node_conf, node_pconf, node_tconf] 
    M_edge = [edge_conf, edge_pconf, edge_tconf]

    # for node
    node_out = []    
    for tra_input in X_node:      
        x = Input(shape=(tra_input[0]*tra_input[1], tra_input[2], tra_input[3]))
        all_inputs.append(x)
        # print(x.shape)
        y = FCN(nb_residual_unit=2)(x)
        node_out.append(y)

    # FM Fusion
    node_w = [tf.Variable(1.) for i in range(len(node_out))]
    node_out = Add()([node_w[i]*node_out[i] for i in range(len(node_out))])

    #  为 node_out 与 edge_out 拼接做准备
    node_out - Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format='channels_first')(node_out)
    
    # for edge
    edge_out = []
    for edg_input in M_edge:
        x = Input(shape=(edg_input[0]*edg_input[1], edg_input[2], edg_input[3]))
        all_inputs.append(x)
        # EM
        y = Flatten()(x)
        y = Dense(edg_input[0]*2*8*8)(y)
        y  =Reshape(target_shape=(edg_input[0]*2, 8, 8))(y)
        # end of EM
        
        # FCN
        y = FCN(nb_residual_unit=2)(y)
        edge_out.append(y)
    # FM Fusion
    edge_w = [tf.Variable(1.) for i in range(len(edge_out))]
    edge_out = Add()([edge_w[i]*edge_out[i] for i in range(len(edge_out))])
    edge_out = Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format='channels_first')(edge_out)

    # Bridge 
    node_edge_out = Add()([node_out, edge_out])

    all_outputs1 = Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format='channels_first')(node_edge_out)
    all_outputs2 = Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format='channels_first')(node_edge_out)

    # for external
    ext = Input(shape=(external_dim,))
    all_inputs.append(ext)

    ext1 = Dense(64)(ext)
    ext2 = Dense(64)(ext)
    ext1 = Reshape((8, 8))(ext1)
    ext2 = Reshape((8, 8))(ext2)

    all_outputs1 = Multiply()([all_outputs1, ext1])
    all_outputs2 = Multiply()([all_outputs2, ext2])
 
    all_outputs1 = Conv2D(filters=2, kernel_size=(3,3), padding='same', data_format='channels_first')(all_outputs1)
    all_outputs1 = Conv2D(filters=128, kernel_size=(3,3), padding='same', data_format='channels_first')(all_outputs1)
    all_outputs1 = Activation('tanh')(all_outputs1)
    all_outputs2 = Activation('tanh')(all_outputs2)
    all_outs = [all_outputs1, all_outputs2]
    model = Model(inputs=all_inputs, outputs=all_outs)
    model.compile(optimizer='adam', loss='mae', metrics=['mse', 'acc'])
    return model


if __name__=='__main__':

    model = MDL(node_conf=(3,2,8,8),node_tconf=(1,2,8,8),node_pconf=(1,2,8,8),
                 edge_conf=(3,128*3,8,8),edge_tconf=(1,128,8,8),edge_pconf=(1,128,8,8))

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, dpi=400)
