import sys
import os
import random
import shutil

train_path = '/home/meizu/WORK/code/YF_baidu_ML/dataset/flowers/flower_photos/train'
test_path = '/home/meizu/WORK/code/YF_baidu_ML/dataset/flowers/flower_photos/test'
classes = os.listdir(train_path)
for cl in classes:
    files_name = os.listdir(os.path.join(train_path, cl))
    random.shuffle(files_name)
    if not os.path.exists(os.path.join(test_path, cl)):
        os.mkdir(os.path.join(test_path, cl))
    for file_name in files_name[:int(round(len(files_name)*0.2))]:
        shutil.move(os.path.join(train_path, cl, file_name), os.path.join(test_path, cl, file_name))
    print ''