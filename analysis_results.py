import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import os

class Analysis_results():
    '''
    * process for Analysis_results
        1. setting value
            * class_list
            * model
    * function for anaylsis
        - show_predictions (with GUI)
        - show_confusion_matrix (with GUI)
        - saving_results

    * OUTPUT
        - [Analysis_results]show_predictions_[dataset name].png
        - [Analysis_results]show_confusion_matrix_[dataset name].png
        - [dataset name]_class_accuracy.txt
    '''
    def __init__(self, class_list, model):
        self.class_list = class_list
        self.model = model
        self.class_list_len = len(class_list)

    # show sample predictions
    def show_predictions(self, dataset, dataset_name) :
        dataset = dataset.shuffle(7).take(15)
        predictions = self.model.predict(dataset)
                
        num_cols = 5
        num_rows = 3
        plt.figure(figsize = (4*num_cols, 2*num_rows))
        img, label = next(iter(dataset)) 
        for plt_idx in range(15):
            plt.subplot(3,5*2,2*plt_idx+1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.numpy()[plt_idx])
            for class_idx in range(self.class_list_len):
                if label.numpy()[plt_idx][class_idx] == 1:
                    real_label = self.class_list[class_idx]
                    break
            predict_label = self.class_list[np.argmax(predictions[plt_idx])]
            if predict_label == real_label:
                color = 'blue'
            else:
                color = 'red'
            plt.xlabel("{} \n({})".format( 
                predict_label,
                real_label),
                color = color)
            
            plt.subplot(3,5*2, 2*plt_idx+2)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.ylim([0,1])
            '''
            if(color == 'red'):
                plt.xlabel("confused with\n"+predict_label,color=color)
            '''
            bar_label = '[confused classes]\n'
            for i in range(self.class_list_len):
                if(predictions[plt_idx][i] > 0.5) :
                    bar_label += self.class_list[i]+'\n'
            plt.xlabel(bar_label)

            bargraph = plt.bar(range(self.class_list_len),predictions[plt_idx], color="#777777")
            
            bargraph[np.argmax(predictions[plt_idx])].set_color(color)
            bargraph[class_idx].set_color('blue')

        plt.tight_layout()
        plt.show()

        print('\nshow_predictions : saving sample predictions of ' + dataset_name + '...\n')
        plt.savefig('[Analysis_results]show_predictions_'+ dataset_name + '.png')

    # show confusion matrix and report
    def show_confusion_matrix(self, dataset, dataset_name):
        print("\n============ confusion matrix & report (%s) ============"%(dataset_name))
        predictions = self.model.predict(dataset)
        real_labels_temp = [y.numpy().tolist() for x, y in dataset] #sample_fromTest
        real_labels = []
        for i in real_labels_temp:
            for j in i: 
                real_labels.append(j)

        #for checking exact percent of prediction
        percent = 0
        for i in range(len(real_labels)):
            r = np.array(real_labels[i])
            p = np.array(predictions[i])

            if np.argmax(r) == np.argmax(p):
                percent+=1
        #print(percent)
        #print(percent/len(real_labels))

        plt.figure(figsize=(7,7))
        cm = confusion_matrix(np.argmax(real_labels, axis=-1), np.argmax(predictions, axis=-1), normalize='true')
        sns.heatmap(cm,annot=True, fmt='d',cmap='Blues')
        plt.xticks(range(self.class_list_len),self.class_list, rotation=90,fontsize=8)
        plt.yticks(range(self.class_list_len),self.class_list, rotation=45,fontsize=8)
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.show()
        
        print('\nshow_confusion_matrix : saving confusion matrix of ' + dataset_name + '...\n')
        plt.savefig('[Analysis_results]show_confusion_matrix_'+ dataset_name + '.png')

        print(classification_report(np.argmax(real_labels, axis=-1),np.argmax(predictions,axis=-1)))

    #decoding the label
    def decoding_label(self, label):
        return self.class_list[np.argmax(label)]

    def createFolder(self, directory):
        try:
            if os.path.exists(directory):
                for file in os.scandir(directory):
                    os.remove(file.path)
            else :
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

    #saving clasification result
    def saving_results(self, dataset, dataset_name):
        print('\nsaving false img & result img & class accuracy file...')
        self.createFolder('false_img')
        self.createFolder('result_img')
        acc_file = open(dataset_name + "_class_accuracy.txt", 'w')

        categories = "idx\t\t\t"
        for i in self.class_list:
            categories += i + '\t'
        acc_file.write(categories+'\n')

        predictions = self.model.predict(dataset)
        pred_idx = 0
        for batch_img, batch_label in dataset:
            for batch_idx, label in enumerate(batch_label):
                real_label = self.decoding_label(label.numpy())
                predict_label = self.decoding_label(predictions[pred_idx])

                results = str(pred_idx) + "\t" + real_label +"\t:\t"
                
                for value in predictions[pred_idx] :
                    value = round(value,3) * 100
                    results += '%.1f' %value
                    results += '\t'
                
                results += ">> " + predict_label + "\t"
                if real_label == predict_label :
                    correct = '[correct]'
                else :
                    correct = '[wrong]'
                results += correct

                acc_file.write(results+"\n")

                img = batch_img[batch_idx].numpy() * 255
                img = Image.fromarray(img.astype('uint8'),'RGB')
                img_filename = dataset_name + "_" + str(pred_idx) + "_" + real_label
                img.save("result_img/" + img_filename + ".png", 'PNG')
                if correct == '[wrong]' :
                    img.save("false_img/" + img_filename + "_not_" + predict_label + ".png", 'PNG')

                pred_idx += 1

        acc_file.close()
        print('\ncompletely saved\n')

    def top_acc(self, dataset):
        predictions = self.model.predict(dataset)
        pred_idx = 0

        top1_acc = 0
        top2_acc = 0
        for batch_img, batch_label in dataset:
            for batch_idx, label in enumerate(batch_label): 
                real_label = self.decoding_label(label.numpy())
                predict = predictions[pred_idx]

                top1 = self.decoding_label(predict)
                
                if top1 == real_label:
                    top1_acc += 1
                    top2_acc += 1
                
                predict[np.argmax(predict)] = 0
                
                top2 = self.decoding_label(predict)

                if top2 == real_label:
                    top2_acc += 1

                pred_idx += 1
        print('\ntop-1 accuracy : %.2f%%'%( 100. * top1_acc / len(predictions)))
        print('top-2 accuracy : %.2f%%\n'%( 100. * top2_acc / len(predictions)))