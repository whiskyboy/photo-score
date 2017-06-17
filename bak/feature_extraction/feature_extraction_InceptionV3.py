# coding: utf-8
import os
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M')

img_path = "../data/img/"
#img_path = "../data/sample/"
save_path = "../data/pretrained_features/"
if not os.path.exists(save_path):
    os.mkdir(save_path)

batch_size = 64
save_size = 5000 # save features per 10000 images

# create the base pre-trained model
model = InceptionV3(weights='imagenet', include_top=False)

def get_imgid(imgname):
    imgid, extname = os.path.splitext(imgname)
    return int(imgid)

all_files = os.listdir(img_path)
for epoch in range(0, len(all_files), save_size):
    img_ids = []
    imgs = []
    for imgname in all_files[epoch:epoch+save_size]:
        try:
            imgid = get_imgid(imgname)
            filename = img_path+imgname
            img = image.load_img(filename, target_size=(299, 299))
            x = image.img_to_array(img)
            x = preprocess_input(x)
        except Exception, e:
            logging.error(str(e))
            continue
        img_ids.append(imgid)
        imgs.append(x)
    logging.info("extracting feature of image[%d-%d]..."%(epoch, epoch+save_size-1))
    img_ids = np.array(img_ids)
    img_features = model.predict(np.array(imgs), batch_size)
    logging.info("saving to disk...")
    np.savez_compressed(save_path+"imgs_epoch=%d.npz"%(epoch//save_size), 
            img_ids = img_ids,
            img_features = img_features)

logging.info("Done")
