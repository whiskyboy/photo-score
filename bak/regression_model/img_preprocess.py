# coding: utf-8

# directory prepare
import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M')
path = "./data/"
if not os.path.exists(path):
    sys.exit(1)

train_path = path + "train/"
if not os.path.exists(train_path):
    os.mkdir(train_path)
valid_path = path + "valid/"
if not os.path.exists(valid_path):
    os.mkdir(valid_path)
img_path = "../data/img/"


import numpy as np
from PIL import Image

# get raw score for each imgid
LAMBDA = 0.05
def get_raw_score(click, zan_num, cai_num):
    return np.log(click*(zan_num*LAMBDA+1)/(cai_num+1)+1)

def saveimg(imgid, score, trg_dir, trg_shape=(224, 224)):
    src_img = img_path+"%s.jpg"%imgid
    trg_img = trg_dir+"%s_%s.jpg"%(imgid, score)
    try:
        img=Image.open(src_img)
        small_img = img.resize(trg_shape)
        small_img.save(trg_img)
    except Exception, e:
        logging.error("saveimg to %s failed."%trg_img)
        logging.error(str(e))
    else:
        logging.info("saveimg to %s successfully."%trg_img)

# load img attr and score each image
img_score_dict = {}
with open("../data/img_attr.csv", 'r') as fin:
    for line in fin:
        attrs = line.strip().split("\t")
        if len(attrs) != 7:
            continue
        imgid, zan_num, cai_num, click, hotness, date, title = attrs
        if date == "NAN":
            continue
        key = date[:7]
        img_score_dict.setdefault(key, [])
        img_score_dict[key].append((imgid, get_raw_score(int(click), int(zan_num), int(cai_num))))
        
# get noramlize score
SAMPLE = False
sample_key = "2016/10"
for key, img_scores in img_score_dict.items():
    if SAMPLE and key != sample_key:
        continue
    
    imgids = [imgid for imgid, score in img_scores]
    scores = [score for imgid, score in img_scores]
    scores = np.array(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    normalized_score = (scores - mean_score) / std_score
    
    for imgid, score in zip(imgids, normalized_score):
        if np.random.rand() > 0.1:
            # copy this img to train directory
            saveimg(imgid, score, train_path)
        else:
            # copy this img to valid directory
            saveimg(imgid, score, valid_path)

logging.info("Finished.")

