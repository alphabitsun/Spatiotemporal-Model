import tensorflow as tf 
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add, Activation,BatchNormalization,Conv2D, Conv3D, Reshape


def resdual_units():
    def f(input):
        output = BatchNormalization()(input)
        output = Activation('relu')(output)
        output = Conv2D(filters=64, kernel_size=3, padding='same')(output)
        output = BatchNormalization()(input)
        output = Activation('relu')(output)
        output = Conv2D(filters=64, kernel_size=3, padding='same')(output)
        return Add()([input, output])
    return f

def ResUnits(repetations=1):
    def f(input):
        for i in range(repetations):
            input = resdual_units()(input)
        return input
    return f

def ST3DNet(c_conf=(6, 2, 8, 8), t_conf=(4, 2, 8, 8), external_dim=8, nb_residual_unit=1):
    
    mian_inputs = []
    outputs = []

    # for closeness
    len_closeness, nb_flow, map_h, map_w = c_conf
    c_input = Input(shape=(len_closeness, map_h, map_w, nb_flow))
    mian_inputs.append(c_input)
    c_out = Conv3D(filters=64, kernel_size=(6,3,3),strides=(1,1,1), activation='relu', padding='same')(c_input)
    # print(c_out.shape)
    c_out = Conv3D(filters=64, kernel_size=(3,3,3),strides=(3,1,1),activation='relu', padding='same')(c_out)
    c_out = Conv3D(filters=64, kernel_size=(3,3,3),strides=(3,1,1), padding='same')(c_out)

    c_out = Reshape((map_h, map_w, 64))(c_out)
    c_out = ResUnits(repetations=nb_residual_unit)(c_out)
    c_out = Conv2D(filters=2, kernel_size=(3,3), padding='same')(c_out)
    print(c_out.shape)
    outputs.append(c_out)

    # for tendency

    len_tendency, nb_flow, map_h, map_w = t_conf
    t_input = Input(shape=(len_tendency, map_h, map_w, nb_flow))
    mian_inputs.append(t_input)

    t_out = Conv3D(filters=8, kernel_size=(len_tendency,1,1), padding="valid", activation='relu')(t_input)
    t_out = Reshape((map_h, map_w,8))(t_out)
    t_out = Conv2D(filters=2, kernel_size=(3,3), padding='same')(t_out)
    print(t_out.shape)
    
    outputs.append(t_out)

    # Fusion
    main_outputs = Add()([outputs[0], outputs[1]])
    main_outputs = Activation('relu')(main_outputs)
    model = Model(inputs=mian_inputs, outputs = main_outputs)
    return model

if __name__=='__main__':
    model = ST3DNet()
    model.summary()