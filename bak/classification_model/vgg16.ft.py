# coding: utf-8
# a *TODO* list:
# 1. define the network structure
# 2. load pre-trained weight
# 3. modify the network structure for finetune
# 4. define the batch flow for training and validation from directory
# 5. train the model
# 6. make a prediction on test data
# 7. submit the results

# Step 1: define the network structure
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Dropout, Flatten, Lambda
from keras import backend as K
K.set_image_data_format("channels_first")

class_num = 5

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
    
def AddFCBlock(model, units, dropout=0.5):
    """
    Args:
        model: keras sequential model
        units: positive integer, dimensionality of the output space
        dropout: dropout rate
    """
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

vgg_model.summary()

# step2: load pre-trained weight
vgg_model.load_weights("./models/vgg16.h5")

# step3: modify the network structure for finetune
# 重新定义模型

while type(vgg_model.layers[-3]) is not Flatten:
    vgg_model.pop()

for layer in vgg_model.layers:
    layer.trainable = False
# 添加全连接层和dropout
# AddFCBlock(vgg_model, units=4096, dropout=None)
AddFCBlock(vgg_model, units=4096, dropout=None)
vgg_model.add(Dense(class_num, activation="softmax"))

# 编译模型（设定学习算法和参数）
vgg_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

vgg_model.summary()

# step4: define the batch flow for training and validation from directory

# 定义数据根目录（先用sample目标调试程序，真正训练时切换到data目录下）
path = "./sample/"

batch_size = 16 # 定义批处理的数据集大小：较小的batch_size可以增加权重调整的次数，同时节省内存的开销

# 这里要用到一个直接从磁盘目录中流式读取图片的工具类：ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
IDG = ImageDataGenerator()
train_batch = IDG.flow_from_directory(path + "train/", target_size=(224, 224), 
                    class_mode='categorical', batch_size=batch_size, shuffle=True)
valid_batch = IDG.flow_from_directory(path + "valid/", target_size=(224, 224), 
                    class_mode='categorical', batch_size=batch_size, shuffle=True)
#test_batch = IDG.flow_from_directory(path + "test/", target_size=(224, 224), 
#                    class_mode=None, batch_size=batch_size, shuffle=False)

# step5: train this model using fit_generator
vgg_model.fit_generator(train_batch, steps_per_epoch=train_batch.samples // batch_size, epochs=3,
                       validation_data=valid_batch, validation_steps=valid_batch.samples // batch_size)

# save weights
vgg_model.save_weights("./models/vgg16_ft.h5")

# step6: make a prediction on test data using predict_generator
# preds = vgg_model.predict_generator(test_batch, steps=np.ceil(test_batch.samples * 1.0 / batch_size))

# step7: make the submission file and submit the results
# get the prob of being a dog
# dog_idx = train_batch.class_indices["dogs"]
# isdog_pred = preds[:, dog_idx]

# get the file-id of each test sample
# file_ids = map(lambda f: int(f[8:f.find(".")]), test_batch.filenames)

# join the two columns into an array of [imageId, isDog]
# subm = np.stack([file_ids, isdog_pred], axis=1)

# save the result
# np.savetxt(path + "results/submission.csv", subm, fmt='%d,%.5f', header='id,label', comments='')
