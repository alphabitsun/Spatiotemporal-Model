"""
    Attentive Traffic Flow Machines

"""

import tensorflow as tf 
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add, Concatenate, Conv2, ConvLSTM2D


def ATFM():

    # S_in:(n, h, w) 当前时刻的前 n 个时刻数据   -->  S_f:(c, h, w)
    # P_in:(m, h, w) 前 m 天同一时刻的数据 -->  P_f:(c, h, w)


    # step 1:
     
    
    model = Model()
    return model