import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
from utils_ps import split_dataset, stead_2_mydataset, BatchGenerator, lr_reducer, early_stop, training_milestones, BuildModel
import h5py
import multiprocessing
import os, gc
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False



def configure_gpu(gpuid, gpu_limit):
    if gpuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_limit
        session = tf.Session(config=config)
        K.tensorflow_backend.set_session(session)



def create_callback(monitor='val_loss', patience=15, outmodel=None):

    reduce_lr = lr_reducer(monitor=monitor, ptnc=patience)
    early_stopping = early_stop(monitor=monitor, patience=patience)
    checkpoints = training_milestones(outmodel, monitor='val_loss')

    return [checkpoints, reduce_lr, early_stopping]



def make_plot(history, name):

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Performance')
    plt.legend()
    plt.savefig(name, dpi = 500)
    plt.close()




def trainer(maindir = None,
                hffile = None,
                add_noise = False,
                add_noise_c = 0.4,
                inp_length = 300,
                label_size = 1,
                train_valid_test = [0.7, 0.15, 0.15],
                channels = 3,
                shuffle = True,
                cnn_layers = 2,
                dense_layers = 1,
                lstm_layers = 2,
                pool_size = 3,
                cnn_dropout = 0.2,
                lstm_dropout = 0.1,
                dense_dropout = 0.5,
                cnn_filters = [128, 96],
                activation = 'tanh',
                padding = 'same',
                lstm_neurons = [64, 32],
                dense_neurons = [64, 32],
                cnn_filter_size = [4, 4],
                optimizer = Adam,
                loss_type = 'mse',
                epochs = 200,
                batch_size = 200,
                learning_rate = 0.001,
                monitor = 'val_loss',
                patience = 20,
                multi_process = True,
                num_cpu = 10,
                gpu_ID = 0,
                gpu_limit = 0.9           
                ):


    if not os.path.exists(hffile): stead_2_mydataset(stead_hdf5, hffile, inp_length, 0, 200000)
    configure_gpu(gpu_ID, gpu_limit)
    if not num_cpu:
        num_cpu = multiprocessing.cpu_count() - (multiprocessing.cpu_count() // 2)

    model_args = {'inp_length': inp_length,
                 'channels': channels,
                 'label_size': label_size,
                 'cnn_layers': cnn_layers,
                 'lstm_layers': lstm_layers,
                 'pool_size': pool_size,
                 'dense_layers': dense_layers,
                 'cnn_dropout': cnn_dropout,
                 'lstm_dropout': lstm_dropout,
                 'dense_dropout': dense_dropout,
                 'cnn_filters': cnn_filters,
                 'activation': activation,
                 'padding': padding,
                 'lstm_neurons': lstm_neurons,
                 'dense_neurons': dense_neurons,
                 'cnn_filter_size': cnn_filter_size,
                 'optimizer': optimizer,
                 'loss': loss_type,
                 'epochs': epochs,
                 'batch_size': batch_size,
                 'learning_rate': learning_rate,
                 'monitor': monitor,
                 'patience': patience}


    ds = h5py.File(hffile, 'r')
    datalist = list(ds['data/'])
    ds.close()
    train_data, valid_data = split_dataset(datalist, train_valid_test, maindir)

    train_generator_args = {'maindir': maindir,
                           'hffile': hffile,
                           'batch_size': batch_size,
                           'add_noise': add_noise,
                           'add_noise_c': add_noise_c,
                           'inp_length': inp_length,
                           'channels': channels,
                           'shuffle': shuffle,
                           }


    training_data = BatchGenerator(train_data, **train_generator_args)

    valid_generator_args = {'maindir': maindir,
                           'hffile': hffile,
                           'batch_size': batch_size,
                           'add_noise': False,
                           'add_noise_c': 0,
                           'inp_length': inp_length,
                           'channels': channels,
                           'shuffle': False,
                           }
    validation_data = BatchGenerator(valid_data, **valid_generator_args)


    callback_info = create_callback(monitor=monitor, 
                                    patience=patience, 
                                    outmodel=f'OBSMAG_2_2_1_12896_6432_0.0_0.1_tanh_[44]_3_{loss_type}_dist.h5'
                                    )
    inp_p = Input(shape=(inp_length, channels), name='input_P')
    inp_s = Input(shape=(inp_length, channels), name='input_S')
    dists = Input(shape=(1,), name='distances')
    model = BuildModel(**model_args)(inp_p, inp_s, dists)

    history = model.fit_generator(generator=training_data,
                                    validation_data=validation_data,
                                    use_multiprocessing=multi_process,
                                    workers=num_cpu,    
                                    callbacks=callback_info, 
                                    epochs=epochs)

    make_plot(history, f'OBSMAG_2_2_1_12896_6432_0.0_0.1_tanh_[44]_3_{loss_type}_dist.jpg')
    gc.collect()




stead_hdf5 = '../merged.hdf5'
train = trainer(maindir = './',
                hffile = 'dataset.h5',
                )
train
