import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from tqdm import tqdm
from data_util import load_Dataset_Original
from model_simsiam import get_encoder, get_predictor, train_step



def train_simsiam(f, h, dataset_one, dataset_two, optimizer, epochs=200):
    """
    训练函数
    """
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in tqdm(range(epochs)):
        for ds_one, ds_two in zip(dataset_one, dataset_two):
            loss = train_step(ds_one, ds_two, f, h, optimizer)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(tf.reduce_mean(step_wise_loss).numpy())

        if epoch % 2 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return epoch_wise_loss, f, h



def Step_Original(
    task,
    model_dir, 
    load_model_name,
    output_dir, 
    save_model_name,
    epochs, 
    batch_size, 
    predict_model_name,
    save_predict_model_name
    ):
    """
    训练函数
    """

    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    
    
    dataset_one = load_Dataset_Original(
        task_id =  task,
        batch_size = batch_size, 
        is_oneD_Fourier = False,
        pattern = 'train'
        )
    dataset_two = load_Dataset_Original(
        task_id =  task,
        batch_size = batch_size, 
        is_oneD_Fourier = True,
        pattern = 'train'
        )

    
    get_encoder(codesize=30).summary()
    get_predictor(codesize=30).summary()

    
    decay_steps = 1000
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.01, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6)

    
    f = get_encoder(codesize=30)
    h = get_predictor(codesize=30)

    
    if 0 < len(load_model_name) and 0 < len(predict_model_name):    
        print("load model weights...")
        f.load_weights(model_dir + load_model_name)         
        h.load_weights(model_dir + predict_model_name)

    epoch_wise_loss, f, h = train_simsiam(f, h, dataset_one, dataset_two, optimizer, epochs=epochs)
    plt.plot(epoch_wise_loss)

    f.save_weights(join(output_dir, save_model_name))
    h.save_weights(join(output_dir, save_predict_model_name))

    plt.grid()
    plt.savefig('G:/UCL_3Dprinter/LUMP/epoch_wise_loss.png')


def train_Task(task):
    """
    三个实验的训练:
    1.只使用正常数据
    2.正常数据 + 第一个故障
    3.正常数据 + 第一，二个故障
    通过设置task来选取进行的步骤:
    1, 2, 3
    """
    if task not in [1, 2, 3, 4]:
        raise ValueError("Invalid task number. Please specify a valid task.")

    if task == 1:
        Step_Original(
            task=task,
            model_dir ='', 
            load_model_name ='',
            output_dir='G:/UCL_3Dprinter/LUMP/models/Task_1/',
            save_model_name='Task_1.h5',
            predict_model_name = '',
            save_predict_model_name = 'Task_1_Predictor.h5',
            batch_size = 10,
            epochs = 100
        )
    elif task == 2:
        Step_Original(
            task=task,
            model_dir='G:/UCL_3Dprinter/LUMP/models/Task_1/',
            load_model_name='Task_1.h5',
            output_dir='G:/UCL_3Dprinter/LUMP/models/Task_2/',
            save_model_name='Task_2.h5',
            predict_model_name='Task_1_Predictor.h5',
            save_predict_model_name='Task_2_Predictor.h5',
            batch_size = 10,
            epochs = 100
        )
    elif task == 3:
        Step_Original(
            task=task,
            model_dir='G:/UCL_3Dprinter/LUMP/models/Task_2/',
            load_model_name='Task_2.h5',
            output_dir='G:/UCL_3Dprinter/LUMP/models/Task_3/',
            save_model_name='Task_3.h5',
            predict_model_name='Task_2_Predictor.h5',
            save_predict_model_name='Task_3_Predictor.h5',
            batch_size = 10,
            epochs = 100
        )
    elif task == 4:
        Step_Original(
            task=task,
            model_dir='G:/UCL_3Dprinter/LUMP/models/Task_3/',
            load_model_name='Task_3.h5',
            output_dir='G:/UCL_3Dprinter/LUMP/models/Task_4/',
            save_model_name='Task_4.h5',
            predict_model_name='Task_3_Predictor.h5',
            save_predict_model_name='Task_4_Predictor.h5',
            batch_size = 10,
            epochs = 100
        )



if __name__ == '__main__':
    train_Task(task=1)
    
    train_Task(task=2)
    
    train_Task(task=3)
    
    train_Task(task=4)
    print("Success")

