'''
room scene images를 이용하여 인테리어 style을 구분할 수 있는 classifier

* with GUI tag가 붙어 있는 함수는 shell 환경에서 제대로 작동하지 않을 수 있다.
'''

from data_preperation import Data_preperation
from training_classifier import Classifier
from analysis_results import Analysis_results

def preperation(data_dir):
    dataset = Data_preperation()
    dataset.data_pipeline(data_dir)

    dataset.print_class_list()
    dataset.checking_data() #with GUI
    
    dataset.mapping_dataset()
    dataset.slicing_dataset(train=0.65, val=0.15, test=0.2)
    dataset.checking_slicing_balance() #with GUI

    dataset.prefetching_dataset()

    class_list = dataset.class_list
    train_ds = dataset.train_ds
    test_ds = dataset.test_ds
    val_ds = dataset.val_ds
    
    return class_list, train_ds, test_ds, val_ds

def train_test(class_list, train_ds, val_ds, test_ds, epochs, option):
    classifier = Classifier()
    classifier.aug_layer()
    classifier.layers(len(class_list),option=option) 

    classifier.training(train_ds, val_ds, epochs)
    classifier.accuracy_loss_graph() #with GUI
    
    classifier.testing(test_ds)

    return classifier.model


def classify(data_dir, epochs, option):
    class_list, train_ds, test_ds, val_ds = preperation(data_dir)
    model = train_test(class_list, train_ds, val_ds, test_ds, epochs, option)

    analysis = Analysis_results(class_list, model)

    analysis.show_predictions(test_ds,"test_dataset") #with GUI
    analysis.show_confusion_matrix(test_ds,"test_dataset") #with GUI
    analysis.saving_results(test_ds,"test_dataset")
    analysis.top_acc(test_ds)




if __name__ == '__main__' :
    # option # : 0 CNN / 1 VGG-16 / 2 ResNet-50 / 3 EfficientNet-B0
    classify(data_dir="/home/archidraw/workspace_sumin/dataset_image_pinterest_ver2", epochs=40, option = 3) # pretrained 40 CNN 400
