import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from model_simsiam import get_encoder
from data_util import  get_Data_By_Label, MatHandler
import os
from os.path import join
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.lscp import LSCP
from pyod.models.iforest import IForest
from scipy.spatial.distance import pdist,squareform

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



def knn_classification(train_features, train_labels, test_features, k=3):
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    return predictions



def intra_class_distance(train_features, test_features):
    
    dists = pdist(train_features, metric='euclidean')
    dist_matrix = squareform(dists)
    
    
    matrix = np.sort(dist_matrix, 0)
    min_dist_all = matrix[1]
    dist_temp = np.mean(min_dist_all) + 3 * np.std(min_dist_all)
    
    
    min_dists = []
    for i in range(test_features.shape[0]):
        x_test = test_features[i].reshape(1, -1)
        dist_list = np.sqrt(np.sum(np.square(x_test - train_features), axis=1))
        dist_min = np.min(dist_list)
        min_dists.append(dist_min)
        
    
    pre_labels = np.array(min_dists) < dist_temp
    pre_labels = pre_labels.astype(int)
    
    return pre_labels




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


def feature_extraction(task):
    """
    测试函数
    """

    if task == 1:
        train_list = [0]
        test_list = [0, 1, 2, 3]
        output_dir = 'TII_method/models/Task_1'
        save_model_name = 'Task_1.h5'
    elif task == 2:
        train_list = [0, 1]
        test_list = [1, 2, 3]
        output_dir = 'TII_method/models/Task_2'
        save_model_name = 'Task_2.h5'
    elif task == 3:
        train_list = [0, 1, 2]
        test_list = [2, 3]
        output_dir = 'TII_method/models/Task_3'
        save_model_name = 'Task_3.h5'
    

    else:
        raise ValueError("Invalid task number. Please specify a valid task.")

    
    weights_path = join(output_dir, save_model_name)

    
    model = load_trained_model(weights_path)

    
    test_data, test_labels = get_Data_By_Label(
        mathandler=MatHandler(is_oneD_Fourier=False),
        pattern='test',
        label_list=test_list
    )


    test_features = extract_features(model, test_data)
    df = pd.DataFrame(test_features)
    df.insert(0, 'label', test_labels)
    
    df.to_csv('TII_method/results/test_data_{}.csv'.format(task), index=False)


    train_data, train_labels = get_Data_By_Label(
        mathandler=MatHandler(is_oneD_Fourier=False),
        pattern='train',
        label_list=train_list
    )

    
    train_features = extract_features(model, train_data)
    df = pd.DataFrame(train_features)
    df.insert(0, 'label', train_labels)
    
    df.to_csv('TII_method/results/train_data_{}.csv'.format(task), index=False)





def test_Task(task, method):

    train_file = pd.read_csv('TII_method/results/train_data_{}.csv'.format(task), header=0)
    train_features = train_file.iloc[-30:, 1:-1].values
    train_labels = train_file.iloc[-30:, 0].values
    
    test_features = pd.read_csv('TII_method/results/test_data_{}.csv'.format(task), header=0, usecols=range(1, 30)).values

    if method == 'KNN':
    
        predicted_labels = knn_classification(train_features, train_labels, test_features)
    elif method == 'distance':
        predicted_labels = intra_class_distance(train_features,test_features)
    elif method == 'DeepSVDD':
        clf = DeepSVDD(epochs=5, contamination=0.01)
        clf.fit(train_features)
        predicted_labels = clf.predict(test_features)
    elif method == 'ECOD':
        clf = ECOD(contamination=0.01)
        clf.fit(train_features)
        predicted_labels = clf.predict(test_features)
    elif method == 'LSCP':
        detector_list = [IForest(max_features=3, n_estimators=40), IForest(max_features=4, n_estimators=40), IForest(max_features=5, n_estimators=50), IForest(max_features=2, n_estimators=50)]
        clf = LSCP(detector_list, contamination=0.01)
        clf.fit(train_features)
        predicted_labels = clf.predict(test_features)
    

    
    file = open('TII_method/results/results.txt', 'a', encoding='utf-8')
    file.write("method: {}, task: {}, pred_labels: {}\n".format(method, task, predicted_labels))





if __name__ == '__main__':

    
    for method in ['distance', 'DeepSVDD', 'ECOD', 'LSCP']:
        test_Task(1, method)
        test_Task(2, method)
        test_Task(3, method)
    print("Success")



