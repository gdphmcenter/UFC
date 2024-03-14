import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from model_simsiam import get_encoder
from data_util import  get_Data_By_Label, MatHandler
import os
from os.path import join


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def load_trained_model(weights_path):
    
    model = get_encoder(codesize=30)
    model.load_weights(weights_path)
    return model


def extract_features(model, data):
    
    features = model.predict(data)
    return features


def load_test_data():
    
    test_data, test_labels = get_Data_By_Label(
        mathandler=MatHandler(is_oneD_Fourier=True),
        pattern='test',
        label_list=[0, 1, 2, 3]
    )
    return test_data, test_labels



def knn_classification(train_features, train_labels, test_features, k=5):
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    return predictions



def intra_class_distance_classification(train_features, train_labels, test_features):
    predicted_labels = []
    
    for test_sample in test_features:
        min_distance = float('inf')
        predicted_label = None
        
        for class_label in np.unique(train_labels):
            class_samples = train_features[train_labels == class_label]
            distances = np.linalg.norm(class_samples - test_sample, axis=1)
            sorted_distances = np.sort(distances)
            threshold_distance = sorted_distances[int(0.95 * len(sorted_distances))]
            
            if threshold_distance < min_distance:
                min_distance = threshold_distance
                predicted_label = class_label
        
        predicted_labels.append(predicted_label)
    predicted_labels = np.array(predicted_labels)
    return predicted_labels



def evaluate_accuracy(predictions, true_labels, label_list):
    
    accuracy_dict = {}
    test_number_dict = {}

    for label in label_list:
        indices = np.where(true_labels == label)[0]
        label_predictions = predictions[indices]
        label_true_labels = true_labels[indices]
        accuracy = np.mean(label_predictions == label_true_labels)
        test_number = len(indices)
        accuracy_dict[label] = accuracy
        test_number_dict[label] = test_number

    return accuracy_dict, test_number_dict



def test_Task(task):
    """
    测试函数
    """

    if task == 1:
        label_list = [0, 1]
        output_dir = 'G:/UCL_3Dprinter/LUMP/models/Task_1'
        save_model_name = 'Task_1.h5'
    elif task == 2:
        label_list = [0, 1, 2]
        output_dir = 'G:/UCL_3Dprinter/LUMP/models/Task_2'
        save_model_name = 'Task_2.h5'
    elif task == 3:
        label_list = [0, 1, 2, 3]
        output_dir = 'G:/UCL_3Dprinter/LUMP/models/Task_3'
        save_model_name = 'Task_3.h5'
    elif task == 4:
        label_list = [0, 1, 2, 3]
        output_dir = 'G:/UCL_3Dprinter/LUMP/models/Task_4'
        save_model_name = 'Task_4.h5'
    else:
        raise ValueError("Invalid task number. Please specify a valid task.")

    
    weights_path = join(output_dir, save_model_name)

    
    model = load_trained_model(weights_path)

    
    test_data, test_labels = get_Data_By_Label(
        mathandler=MatHandler(is_oneD_Fourier=True),
        pattern='test',
        label_list=label_list
    )

    
    test_features = extract_features(model, test_data)

    
    train_data, train_labels = get_Data_By_Label(
        mathandler=MatHandler(is_oneD_Fourier=True),
        pattern='train',
        label_list=label_list
    )

    
    train_features = extract_features(model, train_data)

    
    predicted_labels = intra_class_distance_classification(train_features, train_labels, test_features)
    accuracy_dict, test_number_dict = evaluate_accuracy(predicted_labels, test_labels, label_list)

    
    predicted_labels = knn_classification(train_features, train_labels, test_features)
    accuracy_dict, test_number_dict = evaluate_accuracy(predicted_labels, test_labels, label_list)


    file = open('G:/UCL_3Dprinter/LUMP/results/results.txt', 'a', encoding='utf-8')
    for label in label_list:
        accuracy = accuracy_dict[label]
        test_number = test_number_dict[label]
        file.write("Task: {}, Class: {}, Test Samples: {}, Accuracy: {:.2f}%\n".format(task, label, test_number, accuracy * 100))





if __name__ == '__main__':

    test_Task(1)
    test_Task(2)
    test_Task(3)
    test_Task(4)
    print("Success")



