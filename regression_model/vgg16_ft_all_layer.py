# coding: utf-8

# Step 1: define the network structure
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Dropout, Flatten, Lambda
from keras import optimizers
from keras.preprocessing.image import *
import os, sys
from keras import backend as K
K.set_image_data_format("channels_first")

def preprocess(img):
    """
    subtract average pixes of each channel
    and reverse the channel axies from 'rgb' to 'bgr'
    Args:
        img: (batch_size, channel_size, height, width)
    """
    vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
    return (img - vgg_mean)[:, ::-1] # 注意第一个维度是batch_size

def AddConvBlock(model, layers, filters):
    """
    Args:
        model: keras model
        layers: number of padding + conv layers
        filters: number of filters
    """
    for _ in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
def AddFCBlock(model, units, dropout=0.5, activation="relu"):
    """
    Args:
        model: keras sequential model
        units: positive integer, dimensionality of the output space
        dropout: dropout rate
    """
    model.add(Dense(units, activation=activation))
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
AddFCBlock(vgg_model, units=4096, dropout=None, activation="relu")
AddFCBlock(vgg_model, units=4096, dropout=None, activation="relu")
vgg_model.add(Dense(1, activation="linear"))

vgg_model.summary()

# step2: load pre-trained weight
# 定义数据根目录（先用sample目标调试程序，真正训练时切换到data目录下）
path = "./data/sample/"

#vgg_model.load_weights(path+"results/vgg16_ft_epoch99.h5")

# step4: define the batch flow for training and validation from directory

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
        """For python 2.x.
        # Returns
            The next batch.
        """
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


# 定义批处理的数据集大小：较小的batch_size可以增加权重调整的次数，同时节省内存的开销
batch_size = 16

# 图片预处理工具类
IDG = ImageDataGenerator()

# 获取自定义label的lambda函数。注意传入的参数是图片文件名
ODG = lambda img_filename: os.path.splitext(img_filename)[0].split("_")[1]

# 从目录文件中流式读取数据，避免训练中一次性加载爆内存
train_batch = CustomDirectoryIterator(path + "train/", IDG, ODG, 
                                      target_size=(224, 224), batch_size=batch_size, shuffle=True)
valid_batch = CustomDirectoryIterator(path + "valid/", IDG, ODG,
                                      target_size=(224, 224), batch_size=batch_size, shuffle=True)

# step5: train this model using fit_generator

# 编译模型（设定学习算法和参数）
if sys.argv[1] == "sgd":
    optimizer = optimizers.SGD(lr=1e-5, decay=1e-4, momentum=0., nesterov=False)
elif sys.argv[1] == "adagrad":
    optimizer = optimizers.Adagrad(lr=1e-5, epsilon=1e-08, decay=0.0)
elif sys.argv[1] == "rmsprop":
    optimizer = optimizers.RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)
vgg_model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

for epoch in range(100):
    vgg_model.fit_generator(train_batch, steps_per_epoch=train_batch.samples // batch_size, epochs=1,
                       validation_data=valid_batch, validation_steps=valid_batch.samples // batch_size)
    # save weights
    vgg_model.save_weights(path+"results/vgg16_ft_all_layer_epoch%d.h5"%epoch)

