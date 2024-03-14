import random
import tensorflow as tf
import numpy as np
import os
import csv

class MatHandler(object):
    def __init__(self, is_oneD_Fourier):
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset(is_oneD_Fourier)


    def read_csv(self):
        
        data = np.array([],[])
        label = np.array([])
        count = 0
        
        for fn in os.listdir('G:/UCL_3Dprinter/oneD/'):
            if fn.endswith('.csv'):
                path = 'G:/UCL_3Dprinter/oneD/'+"".join(fn)   
                read_data = csv.reader(open(path, 'r'))
                now_data_label = fn.split('_')[0]
                temp_data = []
                for layer in read_data:
                    temp_data.append(layer)
                
                
                temp_data = np.array(temp_data)
                now_data = temp_data[:,2]           
                now_data = now_data.astype(np.float64)     

                
                now_data = now_data.reshape(-1,1024) 
                now_data_len = now_data.shape[0]

                
                for layer in range(int(now_data_len)):
                    label = np.append(label, int(now_data_label))
                
                
                if count == 0:
                    data = now_data
                    count += 1
                    continue
                
                data = np.vstack((data,now_data))
                count += 1
        
        
        data = data.reshape(-1, 1024, 1)
        return data, label



    def load_dataset(self, is_oneD_Fourier):
        X, y = self.read_csv()

        class_labels = np.unique(y)
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []

        for label in class_labels:
            indices = np.where(y == label)[0]
            random.shuffle(indices)  
            X_train.extend(X[indices[:30]])
            y_train.extend(y[indices[:30]])  
            X_val.extend(X[indices[30:40]])  
            y_val.extend(y[indices[30:40]])
            X_test.extend(X[indices[40:]])   
            y_test.extend(y[indices[40:]])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        if is_oneD_Fourier:
            X_train = oneD_Fourier(X_train)
            X_val = oneD_Fourier(X_val)
            X_test = oneD_Fourier(X_test)
        return X_train, y_train, X_val, y_val, X_test, y_test



def add1(x):
    """
    不做任何处理
    """
    return x


def oneD_Fourier(data):
    """
    一维傅里叶变换
    """
    data = np.squeeze(data)
    for layer in range(data.shape[0]):
        data[layer] = np.abs(np.fft.fft(data[layer]))
    data = data.reshape(-1, 1024, 1)
    return data


def get_Data_By_Label(mathandler, pattern='train', label_list=[0, 1, 2, 3]):
    """
    通过标签获得数据集
    """
    if pattern == 'train':
        data = mathandler.X_train
        label = mathandler.y_train
    elif pattern == 'knn':
        data = mathandler.X_val
        label = mathandler.y_val
    else:
        data = mathandler.X_test
        label = mathandler.y_test

    data_selected = []
    label_selected = []

    for i in label_list:
        idx = np.where(label == i)[0]
        data_temp = data[idx]
        label_temp = label[idx]
        data_selected.append(data_temp)
        label_selected.append(label_temp)

    data_selected = np.concatenate(data_selected)
    label_selected = np.concatenate(label_selected)

    return data_selected, label_selected



def data_fft_real(signal):
    fft = np.fft.fft(signal)
    fft_real = abs(fft)
    return fft, fft_real


def add_pseudo_data(X_train, X_memory, is_oneD_Fourier=True, weight=0.5):
        pseudo_X_train = []
        for i in range(X_train.shape[0]):
            fft_memory_data, _ = data_fft_real(X_memory[i])
            fft_original_subset_data, _ = data_fft_real(X_train[i])
            pseudo_x = weight * fft_memory_data + (1 - weight) * fft_original_subset_data  
            ifft_pseudo_x = np.fft.ifft(pseudo_x)
            ifft_pseudo_x_new = np.concatenate((ifft_pseudo_x.real, ifft_pseudo_x.imag))
            ifft_pseudo_x_new = ifft_pseudo_x_new[:1024]  
            pseudo_X_train.append(ifft_pseudo_x_new)
        if is_oneD_Fourier:
            X_train = oneD_Fourier(X_train)
        return pseudo_X_train



def load_Dataset_Original( task_id=1, batch_size=1, is_oneD_Fourier=True, pattern='train'):

    if task_id == 1:
        data, labels = get_Data_By_Label(
        mathandler=MatHandler(is_oneD_Fourier=is_oneD_Fourier),
        pattern='train',
        label_list=[0]  
    )
    elif task_id == 2:
        X_train, labels = get_Data_By_Label(mathandler=MatHandler(is_oneD_Fourier=False),pattern='train',label_list=[1] ) 
        X_memory, labels = get_Data_By_Label(mathandler=MatHandler(is_oneD_Fourier=False),pattern='train',label_list=[0] ) 
        data = add_pseudo_data(X_train, X_memory, is_oneD_Fourier=is_oneD_Fourier, weight=0.5)
    elif task_id == 3:
        X_train, labels = get_Data_By_Label(mathandler=MatHandler(is_oneD_Fourier=False),pattern='train',label_list=[2] ) 
        X_train = np.squeeze(X_train)
        X_memory, memory_labels = get_Data_By_Label(mathandler=MatHandler(is_oneD_Fourier=False),pattern='train',label_list=[0,1] ) 
        selected_samples = []
        unique_labels = np.unique(memory_labels)
        for label in unique_labels:
            label_indices = np.where(memory_labels == label)[0]
            selected_indices = np.random.choice(label_indices, size=15, replace=False)
            a = X_memory[selected_indices]
            data_temp = np.squeeze(a)
            selected_samples.append(data_temp)
        selected_memory = np.concatenate(selected_samples)
        data = add_pseudo_data(X_train, selected_memory, is_oneD_Fourier=is_oneD_Fourier, weight=0.5)
    elif task_id == 4:
        X_train, labels = get_Data_By_Label(mathandler=MatHandler(is_oneD_Fourier=False),pattern='train',label_list=[3] ) 
        X_train = np.squeeze(X_train)
        X_memory, memory_labels = get_Data_By_Label(mathandler=MatHandler(is_oneD_Fourier=False),pattern='train',label_list=[0,1,2] ) 
        selected_samples = []
        unique_labels = np.unique(memory_labels)
        for label in unique_labels:
            label_indices = np.where(memory_labels == label)[0]
            selected_indices = np.random.choice(label_indices, size=10, replace=False)
            a = X_memory[selected_indices]
            data_temp = np.squeeze(a)
            selected_samples.append(data_temp)
        selected_memory = np.concatenate(selected_samples)
        data = add_pseudo_data(X_train, selected_memory, is_oneD_Fourier=is_oneD_Fourier, weight=0.5)

    AUTO = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(1024).map(add1, num_parallel_calls=AUTO).batch(batch_size).prefetch(AUTO)

    return dataset





if __name__ == "__main__":
    """
    测试数据集的生成效果
    """


    data, label = get_Data_By_Label(label_list=[])       
    print(data)
    print(type(data))
    print(label)
    print(data.shape)
    print(label.shape)
    print('suc')
