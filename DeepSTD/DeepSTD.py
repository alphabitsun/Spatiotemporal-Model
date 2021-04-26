import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add, Dense, Multiply, Conv2D, Conv3D, Concatenate, Flatten, Subtract, Reshape

def IIF():
    # Inherent Influence Factor
    # for POI
    # shape = (I,J,K)
    def f(input):

        # # frequency = [i,j,k] / sum([i,j,:])
        # frequency = input
        # s = tf.reduce_sum(frequency, axis=3)
        # frequency = tf.divide(frequency, s)

        # # density = [i,j,k] / sum([:,:,k])
        # density = input
        # for k in range(density.shape[-1]):
        #     s = sum(density[:,:,k])
        #     density[:,:,k] = density[:,:,k] / s

        # # IBD: imbalance degree
        # # Shannon entropy
        # # S(k) = -ΣΣ(Den(i,j,k)*logDen(i,j,k))
        # # S_max = log(I * J)
        # # IDB(k) = 1 - S(k) / S_max
        # ibd = density
        # S = []
        # ibd = Multiply()(ibd, tf.math.log(ibd))
        # ibd = tf.reduce_sum(ibd, axis=(0,1))
        # S_max = tf.math.log(input.shape[1]*input.shape[2])
        # ibd = tf.divide(ibd, S_max)
        
        # iif = Multiply()([frequency, density, ibd])
        # return iif   # shape=(I,J,K)
        return input
    return f

def DIF(k=32):
    # Disturbance Influence Factor
    # for Multiple Context Factors 
    # time and weather
    # shape = (n+1,k)
    def f(input): 
        # two fully-connected layers
        # out = Flatten()(input)
        out = Dense(128, activation='relu')(input)
        out = Dense(k)(out)
        # out = Reshape((5, k))(out)
        return out
    return f


def ResUnit(filters=64, kernel_size=(3,3)):
    def f(input):
        output = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(input)
        output = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(output)
        return Add()([input, output])
    return f
def ResNet(repetition=2):
    def f(input):
        input = Conv2D(filters=64, kernel_size=(3,3), padding='same')(input)
        for i in range(repetition):
            input = ResUnit(filters=64, kernel_size=(3,3))(input)
        
        output = Conv2D(filters=1, kernel_size=(3,3), padding='same')(input)
        return output
    return f

def DeepSTD(history=(32,32,4), poi=(32,32,32), context=(5, 18)):
    # input:
    #   POI:    p_inputs=(I,J,K) 
    #   Multiple Context Factors: c_inputs=(C,N+1) 
    #   Traffic Flow: t_inputs=(I,J,N)

    main_inputs = []

    p_inputs = Input(shape=(poi), name='POI_Inputs')
    main_inputs.append(p_inputs)
    p_outputs = IIF()(main_inputs[0])
    print(p_outputs.shape) # (32,32,32)

    c_inputs = Input(shape=(context), name='Context_Inputs')
    main_inputs.append(c_inputs)
    c_outputs = DIF(k=poi[-1])(main_inputs[1])
    print(c_outputs.shape)

    # two-step fusion method to fuse IIF and DIF -> IDIF
    # IIF: (I,J,K)
    # DIF: (N+1,K)
    # IDIF: (N+1,I,J,K)
    IDIF = []
    # c_outputs = Reshape((5, 32, c_outputs.shape[0]))(c_outputs)
    for i in range(context[0]):
        IDIF.append(Multiply()([p_outputs, Reshape((32,))(c_outputs[:,i:i+1])]))
    
    IDIF = Reshape((context[0], 32, 32, 32))(IDIF)
    print(IDIF.shape) #(N+1, I, J, K)
    # 3D CNN * 6  --> STD: (N+1,I,J,1)
    std_out = Conv3D(filters=64, kernel_size=(3,3,3), padding='same')(IDIF)
    std_out = Conv3D(filters=64, kernel_size=(3,3,3), padding='same')(std_out)
    std_out = Conv3D(filters=64, kernel_size=(3,3,3), padding='same')(std_out)
    std_out = Conv3D(filters=64, kernel_size=(3,3,3), padding='same')(std_out)
    std_out = Conv3D(filters=64, kernel_size=(3,3,3), padding='same')(std_out)
    std_out = Conv3D(filters=1, kernel_size=(3,3,3), padding='same')(std_out)
    print('std_out.shape:', std_out.shape)

    t_inputs = Input(shape=(history))
    main_inputs.append(t_inputs)
    res_out = Subtract()([main_inputs[2], Reshape((32,32,4))(std_out[:,:4])])

    res_out = ResNet(repetition=1)(res_out)

    main_out = Add()([std_out[:,4:5], res_out])
    main_out = Reshape((history[0], history[1]))(main_out)
    model = Model(inputs=main_inputs, outputs=main_out) 

    return model


if __name__=="__main__":
    # input = tf.constant(shape=(32,32,3))

    model = DeepSTD()
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True, rankdir='LR')