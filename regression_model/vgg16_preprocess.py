# coding: utf-8
import os
import logging
import numpy as np
import bcolz
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Dropout, Flatten, Lambda
from keras import optimizers
from keras.preprocessing.image import *
from keras import backend as K
K.set_image_data_format("channels_first")

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M')

root_path = "./data/"
batch_size = 256 

# 自定义DirectoryIterator类，可以返回自定义的label
class CustomDirectoryIterator(DirectoryIterator):
    def __init__(self, directory, image_data_generator, output_data_generator, target_size, batch_size, shuffle):
        super(CustomDirectoryIterator, self).__init__(directory, image_data_generator,
                 target_size=target_size, color_mode='rgb',
                 classes=None, class_mode=None,
                 batch_size=batch_size, shuffle=shuffle, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False)
        self.output_data_generator = output_data_generator
    
    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros(current_batch_size , dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = self.output_data_generator(fname)
        return batch_x, batch_y


def vgg_feature_model():

    def preprocess(img):
        vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
        return (img - vgg_mean)[:, ::-1] # 注意第一个维度是batch_size

    def AddConvBlock(model, layers, filters):
        for _ in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    def AddFCBlock(model, units, dropout=0.5):
        model.add(Dense(units, activation='relu'))
        if dropout is not None:
            model.add(Dropout(dropout))

    vgg_model = Sequential()
    # 预处理：这里要指定输入张量的维度。在后面的模块中一般不需要考虑上一层的输入维度，keras会自动计算
    vgg_model.add(Lambda(preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))
    # 添加卷积模块
    AddConvBlock(vgg_model, 2, 64)
    AddConvBlock(vgg_model, 2, 128)
    AddConvBlock(vgg_model, 3, 256)
    AddConvBlock(vgg_model, 3, 512)
    AddConvBlock(vgg_model, 3, 512)
    # 将(channels, height, width)的三维张量打平成(channels * height * width, )的一维张量
    vgg_model.add(Flatten())
    # 添加全连接层和dropout
    AddFCBlock(vgg_model, units=4096, dropout=0.5)
    AddFCBlock(vgg_model, units=4096, dropout=0.5)
    # the last layer: softmax layer
    vgg_model.add(Dense(units=1000, activation="softmax"))
    
    vgg_model.load_weights("./models/vgg16.h5")

    # pop the fc layer and use flatten layer as output for feature extraction
    while type(vgg_model.layers[-1]) is not Flatten:
        vgg_model.pop()

    return vgg_model


def batch_generator(rootpath):
    ODG = lambda img_filename: os.path.splitext(img_filename)[0].split("_")[1]
    IDG = ImageDataGenerator()

    return CustomDirectoryIterator(rootpath, IDG, ODG,
            target_size=(224, 224), batch_size=batch_size, shuffle=False)

def save_array(fname, arr): 
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname): 
    return bcolz.open(fname)[:]

def check_dir(dir): 
    if not os.path.exists(dir): 
        os.mkdir(dir)

def vgg_preprocess(model, mode="train"):
    img_root_path = root_path + "%s/"%mode
    check_dir(img_root_path)
    save_array_root_path = root_path + "results/%s/"%mode
    check_dir(save_array_root_path)

    features = []
    labels = []
    img_batch = batch_generator(img_root_path)
    steps = img_batch.samples // img_batch.batch_size
    for _ in range(steps+1):
        img, label = img_batch.next()
        fea = model.predict_on_batch(img)
        features.extend(fea)
        labels.extend(label)
        logging.info("[step=%d]processed %.2f%% images"%(_+1, (_+1)*100.0/(steps+1)))

    save_array(save_array_root_path+"feature.bc", np.array(features))
    save_array(save_array_root_path+"label.bc", np.array(labels))


if __name__ == "__main__":
    model = vgg_feature_model()

    vgg_preprocess(model, "train")
    logging.info("ALL IMAGES IN TRAIN MODE HAD BEEN PROCESSED.")
    vgg_preprocess(model, "valid")
    logging.info("ALL IMAGES IN VALID MODE HAD BEEN PROCESSED.")

