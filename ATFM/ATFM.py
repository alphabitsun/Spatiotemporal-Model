"""
    ATFM: Attentive Traffic Flow Machines

"""

import tensorflow as tf 
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add,Activation, Concatenate, Conv2D, ConvLSTM2D, Dense, Reshape


def ResUnit(filters=16, kernel_size=(3,3)):
    def f(input):
        output = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(input)
        output = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(output)
        return Add()([input, output])
    return f

def ResNet(filters=16, kernel_size=(3,3), repetition=1):
    def f(input):
        input = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(input)
        for i in range(repetition):
            input = ResUnit(filters=filters, kernel_size=(3,3))(input)
        feature = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(input)
        return feature 
    return f

def ATFM_unit():
    def f(input):
        output = input
        return output
    return f

def ATFM(s=(3,32,32,2), p=(3,32,32,2), e=(3,18), ):

    # S_in:(n, h, w) 当前时刻的前 n 个时刻数据   -->  S_f:(c, h, w)
    # P_in:(m, h, w) 前 m 天同一时刻的数据 -->  P_f:(c, h, w)

    n, c, h, w = s
    m = p[1]

    s_input = Input(shape=s)
    p_input = Input(shape=p)
    e_input = Input(shape=e)
    """
        Feature Extraction
    """
    # 1、Sequential
    s_feature = []  # (n,32,32,16)
    # print(s_input[:,0,:,:,:].shape)
    for t in range(n):
        s_feature.append(ResNet(filters=16, kernel_size=(3, 3), repetition=1)(s_input[:,t,:,:,:]))
    s_feature = tf.convert_to_tensor(s_feature)
    # print(len(s_feature))
    print(s_feature.shape)
    # 2、Periodic
    p_feature = []  # (m,32,32,16)
    for d in range(m):
        p_feature.append(ResNet(filters=16, kernel_size=(3, 3), repetition=2)(p_input[:, d]))
    
    # 3、External
    e_feature = Dense(256)(e_input)
    e_feature = Dense(16*h*w)(e_feature)  #(16,32,32)



    # step 1: Sequential
    


    # step 2: Periodic
    

    # step 3: Fusin
    # S_f + P_f + E_f  -->  

    Dense(512)
    Dense(1)
    r = Activation('sigmoid')()
    M_f = r * S_f + (1-r) * P_f
    M_f = Conv2D(filters=2)
    main_output = Activation('tanh')(M_f)
    
    model = Model(inputs=[], outputs=main_output)
    return model


if __name__=="__main__":
    model = ATFM()
    model.summary()