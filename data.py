from chainer.datasets import tuple_dataset
#import matplotlib.gridspec as gridspec
#import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import shutil
import glob
import os


class ChainerImage:

    def __init__(self, path, n_class):
        self._path = path
        self._n_class = n_class
        self._dir_lst = self._class_dir_lst(path)

    @classmethod
    def load_class_dic(cls, m_path):
        p_split = m_path.split("/")
        f_path = "/".join(p_split[:len(p_split)-1]) + "/class.info"
        fin = open(f_path, "r")
        c_pairs = [s.strip().split() for s in fin.readlines()]
        fin.close()
        return {int(p[0]):p[1] for p in c_pairs}
        
    def training_dataset(self, is_gray, rate=0.8):
        image_data, label_data = self._preprocess_training_data(is_gray)
        train_d,test_d = np.split(image_data,[int(len(image_data)*rate)])
        train_l,test_l = np.split(label_data,[int(len(label_data)*rate)])
        train = tuple_dataset.TupleDataset(train_d,train_l)
        test = tuple_dataset.TupleDataset(test_d,test_l)
        return train,test

    def prediction_dataset(self, is_gray):
        img_path = sorted([f for f in os.listdir(self._path) if ".jpg" in f])
        pil_imgs = [Image.open(self._path+"/"+f) for f in img_path]
        img_data = np.array([self._normalization(img,is_gray) for img in pil_imgs])
        return img_path, img_data

    def _preprocess_training_data(self, is_gray):
        image_data, label_data = [], []
        for data in np.random.permutation(self._pair_path_and_label()):
            img = Image.open(data[0])
            image_data.append(self._normalization(img,is_gray))
            label_data.append(np.int32(data[1]))
        return np.array(image_data),np.array(label_data)

    def _normalization(self, img, is_gray):
        if is_gray:
            return [np.asarray(np.float32(n)/255.0) for n in img.split()]
        else:
            return [np.asarray(np.float32(n)/255.0) for n in img.split()]

    def _pair_path_and_label(self):
        dataset = []
        for cp in self._dir_lst:
            for f in [f for f in os.listdir(self._path+"/"+cp) if ".jpg" in f]:
                dataset.append([self._path+"/"+cp+"/"+f,self._dir_lst.index(cp)])
        return dataset

    def _class_dir_lst(self, path):
        return sorted([p for p in os.listdir(path) if os.path.isdir(path+"/"+p)])

    def _get_n_class(self):
        return self._n_class
    n_class = property(_get_n_class)

    def _get_dir_lst(self):
        return self._dir_lst
    dir_lst = property(_get_dir_lst)


def main():
    return


if __name__ == '__main__':
    main()
