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
        self._dir_lst = self._get_dir_lst(path)
        self._class_dic = self._class_dic(self._dir_lst)

    def training_dataset(self, rate=0.8):
        image_data, label_data = self._preprocess_dataset()
        train_d,test_d = np.split(image_data,[int(len(image_data)*rate)])
        train_l,test_l = np.split(label_data,[int(len(label_data)*rate)])
        train = tuple_dataset.TupleDataset(train_d,train_l)
        test = tuple_dataset.TupleDataset(test_d,test_l)
        return train,test

    def prediction_dataset(self):
        image_data, label_data = self._preprocess_dataset()
        return tuple_dataset.TupleDataset(image_data,label_data)

    def _preprocess_dataset(self):
        image_data, label_data = [], []
        for data in np.random.permutation(self._pair_path_and_label()):
            img = Image.open(data[0])
            rgb = [np.asarray(np.float32(n)/255.0) for n in img.split()]
            image_data.append(rgb)
            label_data.append(np.int32(data[1]))
        return np.array(image_data),np.array(label_data)

    def _pair_path_and_label(self):
        dataset = []
        for cp in self._dir_lst:
            print(cp)
            for f in [f for f in os.listdir(self._path+"/"+cp) if ".jpg" in f]:
                dataset.append([self._path+"/"+cp+"/"+f,self._dir_lst.index(cp)])
        return dataset

    def _get_dir_lst(self, path):
        return sorted([p for p in os.listdir(path) if os.path.isdir(path+"/"+p)])

    def _class_dic(self,dir_lst):
        return {i:d for i,d in enumerate(dir_lst)}

    def _get_n_class(self):
        return self._n_class
    n_class = property(_get_n_class)


def main():
    return


if __name__ == '__main__':
    main()
