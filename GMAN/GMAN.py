import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import add, Conv2D


def _Conv2D(num_unit, activation, use_bias=True):
    def f(inputs):
        outputs = Conv2D(filters=num_unit, kernel_size=1, strides=1,
                         padding='VALID', activation=activation, use_bias=use_bias)(inputs)
        return outputs
    return f

def FC(filters, activations):
    def f(inputs):
        for num_unit, activation in zip(filters, activations):
            inputs = _Conv2D(num_unit, activation)(inputs)
        return inputs
    return f

def STEmbedding(T):
    def f(SE, TE):
        # spatial embedding (207, 64) -> (1, 1, 207, 64)
        SE = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
        SE = FC(filters=[64, 64], activations=[tf.nn.relu, None])(SE)
        # print('SE:', SE.shape)
        # temporal embedding TE:(24, 2)
        dayofweek = tf.one_hot(TE[..., 0], depth=7)
        timeofday = tf.one_hot(TE[..., 1], depth=T)
        TE = tf.concat((dayofweek, timeofday), axis=-1)
        TE = tf.expand_dims(TE, 2)
        TE = FC(filters=[64,64], activations=[tf.nn.relu, None])(TE)
        # print("TE:", TE.shape)

        STE = tf.add(SE, TE)
        return STE
    return f

def SpatialAttention():
    def f(X, STE_P):
        # print("==In SpatialAttention==\n")
        # print(X.shape)
        # print(STE_P.shape)
        X = tf.concat((X, STE_P), axis=-1)
        query = FC(filters=[64], activations=[tf.nn.relu])(X)
        key = FC(filters=[64], activations=[tf.nn.relu])(X)
        value = FC(filters=[64], activations=[tf.nn.relu])(X)
        # print("quary:", query.shape)
        # print(tf.concat(tf.split(query, 8, axis = -1), axis = 0).shape)
        query = tf.concat(tf.split(query, 8, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, 8, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, 8, axis = -1), axis = 0)
        attention = tf.matmul(query, key, transpose_b = True)
        attention /= (8 ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, 8, axis = 0), axis = -1)
        X = FC(filters=[64, 64], activations=[tf.nn.relu, None])(X)
        return X
    return f

def TemporalAttention(mask=True):
    def f(X, STE_P, mask=mask):
        X = tf.concat((X, STE_P), axis = -1)
        # [batch_size, num_step, N, K * d]
        query = FC(filters=[64], activations=[tf.nn.relu])(X)
        key = FC(filters=[64], activations=[tf.nn.relu])(X)
        value = FC(filters=[64], activations=[tf.nn.relu])(X)
        # [K * batch_size, num_step, N, d]
        query = tf.concat(tf.split(query, 8, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, 8, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, 8, axis = -1), axis = 0)
        # query: [K * batch_size, N, num_step, d]
        # key:   [K * batch_size, N, d, num_step]
        # value: [K * batch_size, N, num_step, d]
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))
        # [K * batch_size, N, num_step, num_step]
        attention = tf.matmul(query, key)
        attention /= (8 ** 0.5)
        # mask attention score
        if mask:
            batch_size = tf.shape(X)[0]
            num_step = X.get_shape()[1]
            N = X.get_shape()[2]
            mask = tf.ones(shape = (num_step, num_step))
            mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
            mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
            mask = tf.tile(mask, multiples = (8 * batch_size, N, 1, 1))
            mask = tf.cast(mask, dtype = tf.bool)
            attention = tf.compat.v2.where(
                condition = mask, x = attention, y = -2 ** 15 + 1)
        # softmax   
        attention = tf.nn.softmax(attention, axis = -1)
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, 8, axis = 0), axis = -1)
        X = FC(filters=[64, 64], activations=[tf.nn.relu, None])(X)
        return X
    return f

def gatedFusion():
    def f(HS, HT):
        XS = FC(filters=[64], activations=[tf.nn.relu])(HS)
        XT = FC(filters=[64], activations=[tf.nn.relu])(HT)
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = FC(filters=[64, 64], activations=[tf.nn.relu, None])(H)
        return H
    return f

def STAttBlock():
    def f(X, STE_P):
        # Spatial Attention
        HS = SpatialAttention()(X, STE_P)
        # print("HS Shape", HS.shape)
        # Temporal Attention
        HT = TemporalAttention()(X, STE_P)
        # print("HT Shape", HT.shape)
        # Gated Fusion
        H = gatedFusion()(HS, HT)
        # print("H Shape", H.shape)
        return tf.add(X, H)
    return f

def transformAttention(K=8, d=8):
    def f(X, STE_P, STE_Q):
        query = FC(filters=[64], activations=[tf.nn.relu])(STE_P)
        key = FC(filters=[64], activations=[tf.nn.relu])(STE_Q)
        value = FC(filters=[64], activations=[tf.nn.relu])(X)
        # query: [K * batch_size, Q, N, d]
        # key:   [K * batch_size, P, N, d]
        # value: [K * batch_size, P, N, d]
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        # query: [K * batch_size, N, Q, d]
        # key:   [K * batch_size, N, d, P]
        # value: [K * batch_size, N, P, d]
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))    
        # [K * batch_size, N, Q, P]
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        # [batch_size, Q, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = FC(filters=[64, 64], activations=[tf.nn.relu, None])(X)
        return X
    return f

def GMAN(SE, P, Q, T, L, K, d, bn, is_training):

    # input
    X_input = Input(shape=(12, 207))
    X_output = tf.expand_dims(X_input, -1)
    X_output = FC(filters=[64, 64], activations=[tf.nn.relu, None])(X_output)
    # print(X_output.shape)

    # STE
    TE = Input(shape=(24, 2), dtype=tf.int64)
    STE = STEmbedding(T=T)(SE, TE)
    # print("STE:", STE.shape) # (None, 24, 207, 64)
    STE_P = STE[:, : P]
    STE_Q = STE[:, P:]

    # encoder L*StAtt block
    for _ in range(L):
        X_output = STAttBlock()(X_output, STE_P)
    # transAttr
    X_output = transformAttention()(X_output, STE_P, STE_Q)
    # decoder
    for _ in range(L):
        X_output = STAttBlock()(X_output, STE_Q)

    X_output = FC(filters=[64,1], activations=[tf.nn.relu, None])(X_output)
    X_output = tf.squeeze(X_output, axis=3)
    model = Model(inputs=[X_input, TE], outputs=X_output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == '__main__':

    """ 
        input: num is the length of the data set    for example: 100
        X:      (num, 12, 207)
        SE:     (207, 64)
        TE:     (num, 24, 2)  --(dayofweek, timeofday)
        Y:      (num, 12, 207)

        P：       number of history steps       # 12
        Q：       number of prediction steps        # 12
        T：       one day is divided into T steps
        L：       number of STAtt blocks in the encoder/decoder
        K：       number of attention heads
        d：       dimension of each attention head outputs
    """
    X = tf.random.normal((100, 12, 207))
    SE = tf.random.normal((207, 64))
    TE = np.load('SE_sample.npy')
    Y = tf.random.normal((100, 12, 207))
    
    model = GMAN(SE, P=12, Q=12, T=12*24, L=1, K=8, d=8, bn=False, is_training=True)
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=False, dpi=400,rankdir='LR')
    model.summary()
    model.fit([X, TE], Y, batch_size=10)