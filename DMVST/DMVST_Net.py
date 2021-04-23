import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Flatten, Dense, LSTM, Concatenate, Reshape


def DMVST_Net(
        image_h = 7,
        image_w = 7,
        len_spa_f = 64,     # len of spatial's feature
        seq = 20,           # len of temporal data 
        len_sem_f = 64,     # len of Semantic's feature
        h = 8  # 4小时
        ):

    all_inputs = []
     
    # Spatial View
    spa_inputs = Input(shape=(image_h, image_w, h))
    spa_outputs = []
    all_inputs.append(spa_inputs)

    for i in range(h):  
        spa_input = spa_inputs[:,:,i:i+1]

        # N * CNN  
        output = Conv2D(filters=64, kernel_size=(3,3), padding='same')(spa_input)
        output = BatchNormalization()(output)
        
        # Flatten + Dense
        output = Flatten()(output)
        output = Dense(len_spa_f)(output)

        spa_outputs.append(output)
    spa_outputs = Reshape((len_spa_f, h))(spa_outputs)  


    # Temporal View
    tem_inputs = Input(shape=(seq, h))
    all_inputs.append(tem_inputs)
    st_out = Concatenate(axis=1)([spa_outputs, tem_inputs])   # (seq + len_spa_f, h)   
    st_out = LSTM(64)(st_out) # (64,)

    # Semantic View
    # 相似度在数据处理部分完成，这里仅设置全连接层
    sem_inputs = Input(shape=(len_sem_f))
    all_inputs.append(sem_inputs)
    sem_outs = Dense(10)(sem_inputs)

    outputs = Concatenate(axis=1)([st_out, sem_outs])

    model = Model(inputs=all_inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    # model.summary()
    return model

if __name__=='__main__':
    model = DMVST_Net()
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    model.summary()