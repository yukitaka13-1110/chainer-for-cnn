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


class Images:

    def __init__(self,directory):
        self._directory = directory
        self._original_path = '../Original/'+directory
        self._img_path = '../Image/'+self.__img_dir(directory)
        self._original_f_lst = self.__collect_files()
        self._f_sum = len(self._original_f_lst)
    
    """ remove image until sum < n """
    def remove_img_under_n(self,n):
        f_lst = self.__collect_files()
        if len(f_lst)+self._f_sum < n:
            return
        target_lst = list(set(f_lst)-set(self._original_f_lst))
        random.shuffle(target_lst)
        for f in target_lst[0:len(target_lst)-n+self._f_sum]:
            os.remove(self._img_path+'/'+f)

    """ rotate image for wider """
    def align_direction(self):
        for f in self.__collect_files():
            img = Image.open(self._img_path+'/'+f)
            if max(img.size)==img.size[1]:
                img = img.transpose(Image.ROTATE_270)
                img.save(self._img_path+'/'+f)
    
    """ translate to square """
    def to_square(self):
        f_lst = self.__collect_files()
        for f in f_lst:
            img = Image.open(self._img_path+'/'+f)
            x,y = self.__matrix_for_paste(img)
            canvas = Image.new('RGB',(max(img.size),max(img.size)),(255,255,255))
            canvas.paste(img,(x,y))
            canvas.save(self._img_path+'/'+f)

    """ resize image """
    def resize(self,vpx,hpx):
        f_lst = self.__collect_files()
        for f in f_lst:
            img = Image.open(self._img_path+'/'+f)
            img = img.resize((vpx, hpx))
            img.save(self._img_path+'/'+f)

    """ translate to gray scale """
    def to_gray(self):
        for f in self.__collect_files():
            img = Image.open(self._img_path+'/'+f)
            img = img.convert("L")
            img.save(self._img_path+'/'+f)

    """ calculate repeat num for create image """
    def calc_repeat_num(self,num):
        return int(float(num-1-self._f_sum)/float(self._f_sum*4))+1 if num-1-self._f_sum > 0 else 0
    
    """ create images by zoom,rotate,shear,shift """
    def create_image(self,zoom=False,rotate=False,shear=False,shift=False,repeat=1):
        from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
        for f in self._original_f_lst:
            img = load_img(self._img_path+'/'+f)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            if zoom:
                self.__zoom(img_array,repeat)
            if rotate:
                self.__rotate(img_array,repeat)
            if shear:
                self.__shear(img_array,repeat)
            if shift:
                self.__shift(img_array,repeat)

    def __matrix_for_paste(self,img):
        width,height = img.size
        return (int((height-width)/2),0) if width < height else (0,int(((width-height)/2)))

    def __create(self,datagen, img_array, repeat, process):
        num = random.randint(1,1000)
        g = datagen.flow(img_array, batch_size=1, save_to_dir=self._img_path, save_prefix=process+'_%d_'%num, save_format='jpg')
        for i in range(repeat):
            batch = g.next()

    def __zoom(self,img_array,repeat):
        from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
        datagen = ImageDataGenerator(zoom_range=0.2)
        self.__create(datagen, img_array, repeat, 'zoom')

    def __shift(self,img_array,repeat):
        from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
        datagen = ImageDataGenerator(width_shift_range=0.2)
        self.__create(datagen, img_array, repeat, 'shift')

    def __rotate(self,img_array,repeat):
        from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
        datagen = ImageDataGenerator(rotation_range=180)
        self.__create(datagen, img_array, repeat, 'rotate')

    def __shear(self,img_array,repeat):
        from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
        datagen = ImageDataGenerator(shear_range=0.78)
        self.__create(datagen, img_array, repeat, 'shear')

    def __collect_files(self):
        return [f for f in os.listdir(self._img_path) if '.DS_Store' not in f and '.' in f]

    def __img_dir(self,directory):
        self.__refresh_dir()
        return directory

    def __to_RGB(self):
        for f in [f for f in os.listdir('../Image/'+self._directory) if 'DS_Store' not in f]:
            img = Image.open('../Image/'+self._directory+'/'+f)
            if img.mode != "RGB":
                img = img.convert("RGB")
            name,ext = f.split('.')
            img.save('../Image/'+self._directory+'/'+name+'.jpg')

    def __refresh_dir(self):
        if os.path.isdir('../Image/'+self._directory):
            shutil.rmtree('../Image/'+self._directory)
        shutil.copytree(self._original_path,'../Image/'+self._directory)
        self.__to_RGB()


def main():
    return


if __name__ == '__main__':
    main()
