#coding:gbk
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Dropout, Flatten, Lambda
from keras import optimizers
from keras.preprocessing.image import *
import os, sys
from keras import backend as K
K.set_image_data_format("channels_first")

class Vgg16():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Lambda(self.preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))
        self.add_conv_block(2, 64)
        self.add_conv_block(2, 128)
        self.add_conv_block(3, 256)
        self.add_conv_block(3, 512)
        self.add_conv_block(3, 512)
        self.model.add(Flatten())
        self.add_fc_block(units=4096, dropout=0.5)
        self.add_fc_block(units=4096, dropout=0.5)
        self.model.add(Dense(units=1000, activation="softmax"))

    def preprocess(self, img):
        vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
        return (img - vgg_mean)[:, ::-1] # 注意第一个维度是batch_size
    
    def add_conv_block(self, layers, filters):
        model = self.model
        for _ in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
    def add_fc_block(self, units, dropout=0.5, activation="relu"):
        model = self.model
        model.add(Dense(units, activation=activation))
        if dropout is not None:
            model.add(Dropout(dropout))
    
    def load_weights(self, model_h5):
        self.model.load_weights(model_h5)

    def get_conv_submodel(self):
        layers = self.model.layers
        flatten_layer_idx = [index for index, layer in enumerate(layers)
                                if type(layer) is Flatten][-1]
        conv_layers = layers[:flatten_layer_idx+1]
        return Sequential(conv_layers)

    def get_fc_submodel(self):
        layers = self.model.layers
        flatten_layer_idx = [index for index, layer in enumerate(layers)
                                if type(layer) is Flatten][-1]
        fc_layers = layers[flatten_layer_idx+1:]
        fc_model = Sequential([
            Dense(4096, activation="relu", input_shape=fc_layers[0].input_shape[1:]),
            Dropout(0.5),
            Dense(4096, activation="relu"),
            Dropout(0.5),
            Dense(1000, activation="softmax")
            ])
        for l1, l2 in zip(fc_model.layers, fc_layers):
            l1.set_weights(l2.get_weights())
        return fc_model

if __name__=="__main__":
    vgg_model = Vgg16()
    vgg_model.load_weights("./models/vgg16.h5")
    conv_model = vgg_model.get_conv_submodel()
    fc_model = vgg_model.get_fc_submodel()

    print fc_model.layers[0].input_shape

