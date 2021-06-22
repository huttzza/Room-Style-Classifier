#training_classifier.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
import matplotlib.pyplot as plt

class Classifier():
    '''
    * process for Classifier
        1. setting value
            * IMG SIZE (width, height, channel) (same with Data_preperation)
        2. aug_layer()
        3. layers()
        4. training()
        5. testing

    * function for checking data
        - accuracy_loss_graph (with GUI)

    * OUTPUT
        - model
        - [Classifier]accuracy_loss_graph.png
    '''
    def __init__(self):
        # IMG_SIZE (same with Data_preperation)
        self.IMG_WIDTH = 224
        self.IMG_HEIGHT = 224
        self.IMG_CHANNEL = 3
        
        # epochs
        self.epochs = 40
   
    # making data_augmentation layer ; 'cause dataset is small
    def aug_layer(self):
        self.data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                            input_shape=(self.IMG_HEIGHT, 
                                                                        self.IMG_WIDTH,
                                                                        self.IMG_CHANNEL)),
                layers.experimental.preprocessing.RandomRotation(0.2),
                layers.experimental.preprocessing.RandomZoom(0.2),
                layers.experimental.preprocessing.RandomCrop(self.IMG_HEIGHT,self.IMG_WIDTH,seed=10),
                layers.experimental.preprocessing.RandomContrast(0.2)
            ]
        )

    def layers(self,num_classes,option):
        IMG_HEIGHT, IMG_WIDTH = self.IMG_HEIGHT, self.IMG_WIDTH
        if option == 0: #CNN
            print('\n-----CNN model-----\n')
            lr = 0.0007
            self.model = Sequential([
                self.data_augmentation,
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),

                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Dropout(0.2),
                
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
            
        elif option == 1: #VGG
            print('\n-----VGG model-----\n')
            lr = 0.00001
            vgg16 = VGG16(weights='imagenet',input_shape=(IMG_HEIGHT,IMG_WIDTH,3), include_top=False)
            self.model = Sequential([
                vgg16,
                layers.Dropout(0.5),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dense(1024, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
            
        elif option == 2 : #ResNet
            print('\n-----ResNet model-----\n')
            lr = 0.0000005
            resnet = ResNet50(weights='imagenet',input_shape=(IMG_HEIGHT, IMG_WIDTH,3), include_top=False, classes=5)
            self.model = Sequential([
                resnet,
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(1024, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
        elif option == 3: #EfficientNet
            print('\n-----EfficientNet model-----\n')
            lr = 0.000001
            eff = EfficientNetB0(weights='imagenet',input_shape=(IMG_HEIGHT,IMG_WIDTH,3), include_top=False, classes=5)
            self.model = Sequential([
                eff,
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.BatchNormalization(),
                layers.Dense(1024, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
        else :
            print('\n!!!!!!!!!![option error]!!!!!!!!!!\n')

        opt = optimizers.Adam (lr = lr)
        
        self.model.compile(optimizer=opt,
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                    metrics=['accuracy'])

        print("\n============ model summary ============\n")
        self.model.summary()

    def training(self, train_ds, val_ds, epochs):
        print("\n============ training ============")

        self.epochs = epochs
        with tf.device('/gpu:0'):#'/device:XLA_GPU:0'):
            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs
            )
        print('\ntrain_accuracy %.4f%% \tval_accuracy %.4f%%' %(self.history.history['accuracy'][-1]*100, self.history.history['val_accuracy'][-1]*100))

        self.model.save("room_style_classifier_model.h5")

    
    def accuracy_loss_graph(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        print('\naccuracy_loss_graph : saving accuracy & loss graph...\n')
        plt.savefig('[Classifier]accuracy_loss_graph.png')
    
    def testing(self, test_ds):
        print("\n============ testing ============")
        results = self.model.evaluate(test_ds)
        print('\ntest_accuracy %.4f%%\n' %(results[1]*100))
