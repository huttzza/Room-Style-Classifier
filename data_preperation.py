#data_preperation.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pathlib

class Data_preperation():
    '''
    * dataset structure
        Category 1 
        　　├─ 1.jpg
        　　└─ 2.jpg
        Category 2
        　　├─ 1.jpg
        　　└─ 2.jpg
    * process for Data_preperation
        1. setting value
            * IMG SIZE (width, height, channel)
            * batch size
        2. data_pipleline(data_dir)
        3. mapping_dataset()
        4. slicing_dataset(train, val, test) @ratio of train, val, test dataset
        5. prefetching_dataset()

    * function for checking data
        - print_class_list
        - checking_data
        - checking_slicing_balance

    * OUTPUT
        - class_list
        - train_ds
        - val_ds
        - test_ds
        - [Data_preperation]checking_data.png
        - [Data_preperation]checking_slicing_balance.png
    '''
    def __init__(self):
        #IMG SIZE
        self.IMG_WIDTH = 224
        self.IMG_HEIGHT = 224
        self.IMG_CHANNEL = 3
        
        #batch size
        self.batch_size = 32

        # ========== OUTPUT ==========
        # class name list
        self.class_list = []

        # ========== for inner fuction ==========
        # file path dataset
        self.dataset = []
        # labeled dataset
        self.labeled_ds = []

        # file full path
        self.ds_list = []  
        # file label
        self.ds_label = [] 
        
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    # making data pipleline
    def data_pipeline(self, data_dir):

        print("\n\nmaking data pipeline from ", data_dir,".........\n\n")
        
        data_dir = pathlib.Path(data_dir)

        #dataset(just file path)
        self.dataset = tf.data.Dataset.list_files(str(data_dir/'*/*')) 

        ds_list = [] # file full path
        ds_label = [] # file label

        for f in self.dataset:
            ds_list.append(f.numpy())
            label = tf.strings.split(f, os.sep)[-2]
            ds_label.append(label.numpy())

        #for checking data function(plt img)
        ds_list = [ds_list[i].decode('UTF-8') for i in range(len(ds_list))] 
        self.ds_list = np.asarray(ds_list)

        #for making class list & checking data function(plt img)
        ds_label = [ds_label[i].decode('UTF-8') for i in range(len(ds_label))]
        self.ds_label = np.asarray(ds_label)

        class_list = list(set(self.ds_label))
        self.class_list = np.array(sorted(class_list))
        self.class_list_len = len(self.class_list)


    # print class list
    def print_class_list(self):
        print("\nCLASS LIST : ", self.class_list)
        
        class_count = [0 for i in range(self.class_list_len)]
        
        print("\n============ DATASET ============")
        for f in self.ds_label:
            for c in range(self.class_list_len) :
                if self.class_list[c] == f:
                    class_count[c] += 1
        for i in range(self.class_list_len):
            print("%s : %d "%(self.class_list[i] ,class_count[i]))

    # using plt, checking data well uploaded
    def checking_data(self):
        image = np.array(Image.open(self.ds_list[0]))
        plt.yticks([])
        plt.xticks([])
        plt.xlabel(self.ds_label[0])
        plt.grid(False)
        plt.imshow(image)
        plt.show()
        print('\nchecking_data : making img pipeline dataset is correctly done.\n')
        plt.savefig('[Data_preperation]checking_data.png')
    
    # 3 function definitions for uploading & mapping data
    def get_label(self,file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == self.class_list

    def decode_img(self,img):
        img = tf.image.decode_jpeg(img, channels=3)
        #nomalization : imgs to 0~1 float type
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [self.IMG_HEIGHT, self.IMG_WIDTH])

    def process_path(self,file_path):
        label = self.get_label(file_path)
        label = tf.dtypes.cast(label, tf.int32)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label
    
    # mapping dataset with label
    def mapping_dataset(self):
        self.labeled_ds = self.dataset.map(self.process_path, num_parallel_calls=self.AUTOTUNE)

        for image, label in self.labeled_ds.take(1):
            print("\nImage shape: ", image.numpy().shape)
            print("Label: ", label.numpy())
    
    # slicing datset >>> train, validate, test
    def slicing_dataset(self, train, val, test):

        print("\nnow slcing dataset >>>>> train/val/test.........")
        
        ds_size = self.labeled_ds.cardinality().numpy()

        train_size = int(train * ds_size)
        val_size = int(val * ds_size)
        test_size = int(test * ds_size)

        self.labeled_ds.shuffle(10)
        self.train_ds = self.labeled_ds.take(train_size)
        temp_ds = self.labeled_ds.skip(train_size)
        self.val_ds = temp_ds.take(val_size)
        self.test_ds = temp_ds.skip(val_size)

        print("# of Train dataset \t: ", self.train_ds.cardinality().numpy())
        print("# of Validate dataset \t: ", self.val_ds.cardinality().numpy())
        print("# of Test dataset \t: ", self.test_ds.cardinality().numpy())
    
    # using plt, show class variation of datasets
    def plot_class_variation(self, dataset, title, location) :
        x = np.arange(self.class_list_le)
        class_count = [0 for i in range(self.class_list_le)]
        for f,l in dataset:
            class_count[np.argmax(l)] += 1
            
        #for print exact value of classes
        print('\n[%s]'%(title))
        for i in range(self.class_list_le):
            print("%s : %d "%(self.class_list[i] ,class_count[i]))
        
        plt.bar(x+location,class_count,label=title, width=0.25)
        plt.legend()
        plt.xticks(range(5),self.class_list,rotation=90,fontsize=8)
        plt.tight_layout()

    #checking balance of slicing
    def checking_slicing_balance(self):
        plt.figure(figsize=(5,5))
        self.plot_class_variation(self.train_ds,"train_ds",0.0)
        self.plot_class_variation(self.val_ds,"val_ds",0.25)
        self.plot_class_variation(self.test_ds,"test_ds",0.50)  
        plt.show()
        
        print('\nchecking_slicing_balance : saving class balance graph...\n')
        plt.savefig('[Data_preperation]checking_slicing_balance.png')
    
    def prefetching_dataset(self):
        #prefetching data(imgs)
        self.train_ds = self.train_ds.cache().shuffle(100).prefetch(buffer_size=self.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

        #batch
        self.train_ds = self.train_ds.batch(self.batch_size)
        self.val_ds = self.val_ds.batch(self.batch_size)
        self.test_ds = self.test_ds.batch(self.batch_size)

        print("\nsuccessfully batch the images\n")
