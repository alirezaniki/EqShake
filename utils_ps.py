import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.layers import LSTM, GRU, Conv1D, Activation, Dense, Dropout, Cropping1D, Lambda, \
    Bidirectional, Layer, Attention, MaxPooling1D, BatchNormalization, UpSampling1D, concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import keras
from tqdm import tqdm
import tensorflow as tf
import h5py
import numpy as np



def stead_2_mydataset(inph5, outh5, len=300, noise=0, num=2000):
    '''
                nnnnnnnnnssssssssssssssssssssss
                  noise     +     signal         = len
    '''

    print('preparing the dataset ..')
    inp = h5py.File(inph5, 'r')
    oh5 = h5py.File(outh5, 'w')
    oh5.create_group('data')
    i=1

    prog = tqdm(total=num)
    for item in inp['data']:
        if not item.endswith('_EV'): continue
        meta = inp.get(f'data/{item}')
        smt = meta.attrs['source_magnitude_type']
        if smt != 'ml': continue

        pas = int(meta.attrs['p_arrival_sample'])
        sas = int(meta.attrs['s_arrival_sample'])

        if pas < noise or sas - pas < len or sas + len - noise > 6000: 
            continue
        
        snr = meta.attrs['snr_db']
        tnm = meta.attrs['trace_name']
        stt = meta.attrs['trace_start_time']
        sdd = meta.attrs['source_distance_deg']
        bad = meta.attrs['back_azimuth_deg']
        snr = meta.attrs['snr_db']
        mag = meta.attrs['source_magnitude']
        dep = meta.attrs['source_depth_km']

        npy_p = np.array(meta)[pas-noise:pas+len-noise, :]
        npy_s = np.array(meta)[sas-noise:sas+len-noise, :]
        npy = np.stack((npy_p, npy_s), axis=0)

        ok = True
        if (np.std(npy, axis=0) == 0).any() or np.isnan(npy).any() or np.isinf(npy).any():
            ok = False

        if not ok: continue
        assert npy.shape == (2, len, 3)
        d = oh5.create_dataset(f'data/{tnm}', shape=(2, len, 3), data=npy, dtype=np.float32)
        
        d.attrs['trace_name'] = tnm
        d.attrs['p_arrival_sample'] = pas
        d.attrs['s_arrival_sample'] = sas
        d.attrs['trace_start_time'] = stt
        d.attrs['source_distance_deg'] = sdd
        d.attrs['back_azimuth_deg'] = bad
        d.attrs['snr_db'] = snr
        d.attrs['source_magnitude'] = mag
        d.attrs['source_depth_km'] = dep

        oh5.flush()
        prog.update()
        if i == num: break
        i+=1
        
    
    inp.close()
    oh5.close()




def split_dataset(datalist, train_valid_test, maindir):
    np.random.shuffle(datalist)
    train = datalist[:int(train_valid_test[0]*len(datalist))]
    valid = datalist[int(train_valid_test[0]*len(datalist)):
                        int((train_valid_test[0] + train_valid_test[1])*len(datalist))]
    test = datalist[int((train_valid_test[0] + train_valid_test[1])*len(datalist)):]
    np.save(f'{maindir}/test', test)

    return train, valid




class BatchGenerator(keras.utils.Sequence):

    def __init__(self,
                 datalist,
                 maindir = None,
                 hffile = None,
                 batch_size = None,
                 add_noise = False,
                 add_noise_c = None,
                 inp_length = None,
                 channels = None,
                 shuffle = False,
                 ):
        self.datalist = datalist
        self.maindir = maindir
        self.hffile = hffile
        self.batch_size = batch_size
        self.add_noise = add_noise
        self.add_noise_c = add_noise_c
        self.inp_length = inp_length
        self.channels = channels
        self.shuffle = shuffle
        self.on_epoch_end()  


    def __len__(self):
        if self.add_noise:
            num_batch = 2 * int(np.floor(len(self.datalist) / self.batch_size))
        else:
            num_batch = int(np.floor(len(self.datalist) / self.batch_size))
        
        return num_batch



    def _add_noise(self, data, snr):
        'Modified from https://github.com/smousavi05/EQTransformer'
        
        data_noisy = np.zeros((data.shape))
        if np.random.uniform(0, 1) < self.add_noise_c and all(snr >= 10.0): 
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
            data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
            data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])    
        else:
            data_noisy = data
            
        return data_noisy
    
    

    def __getitem__(self, index):
        if self.add_noise:
            s = index * self.batch_size // 2
            e = (index+1) * self.batch_size // 2
            indexes = self.indexes[s:e]
            indexes = np.append(indexes, indexes)
        else:
            s = index * self.batch_size
            e = (index+1) * self.batch_size
            indexes = self.indexes[s:e]

        batch_list = [self.datalist[j] for j in indexes]
        xp, xs, dists, y = self._batch_generator(batch_list)
        
        assert not np.any(np.isnan(xp).any())
        assert not np.any(np.isinf(xp).any())
        assert not np.any(np.isnan(xs).any())
        assert not np.any(np.isinf(xs).any())

        return {'input_P': np.array(xp), 'input_S': np.array(xs), 'distances': dists}, np.array(y)


    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.datalist))
        if self.shuffle:
            np.random.shuffle(self.indexes)



    def _batch_generator(self, datalist):
        
        x_p = np.zeros((self.batch_size, self.inp_length, self.channels))
        x_s = np.zeros((self.batch_size, self.inp_length, self.channels))
        dist = np.zeros((self.batch_size))
        y1 = np.zeros((self.batch_size))
        dd = h5py.File(self.hffile, 'r')     

        for i, ev in enumerate(datalist):
            meta = dd.get(f'data/{ev}')
            magnitude = float(meta.attrs['source_magnitude'])
            distance = float(meta.attrs['source_distance_deg'])
            snr = meta.attrs['snr_db']

            data_p = np.array(meta)[0,:,:]
            data_s = np.array(meta)[1,:,:] 

            if self.add_noise:
                data_p = self._add_noise(data_p, snr)
                data_s = self._add_noise(data_s, snr)

            x_p[i, :, :] = data_p
            x_s[i, :, :] = data_s
            dist[i] = distance
            y1[i] = magnitude


        dd.close()
        return x_p, x_s, dist, y1.astype('float32')





def lr_reducer(monitor='val_loss', ptnc=20):
    reduce_lr = ReduceLROnPlateau(monitor=monitor, 
                                factor=0.5,
                                cooldown=0,
                                patience=ptnc//3, 
                                verbose=1, 
                                min_lr=0.5e-6)
    return reduce_lr




def early_stop(monitor='val_loss', patience=20):
    early_stopping = EarlyStopping(monitor=monitor, 
                                patience=patience)
    return early_stopping




def training_milestones(output, monitor='val_loss'):
    checkpoint = ModelCheckpoint(filepath=output,
                                    monitor=monitor, 
                                    mode='auto',
                                    verbose=1,
                                    save_best_only=True)
    return checkpoint




class SelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        attention_output = Attention()([inputs, inputs])
        return attention_output    




class BuildModel():
    
    def __init__(self,
                 inp_length = None,
                 channels = None,
                 label_size = None,
                 cnn_layers = None,
                 lstm_layers = None,
                 pool_size = None,
                 dense_layers = None,
                 cnn_dropout = None,
                 lstm_dropout = None,
                 dense_dropout = None,
                 cnn_filters = None,
                 activation = None,
                 padding = None,
                 lstm_neurons = None,
                 dense_neurons = None,
                 cnn_filter_size = None,
                 optimizer = None,
                 loss = None,
                 epochs = None,
                 batch_size = None,
                 learning_rate = None,
                 monitor = None,
                 patience = None
                 ):
        
        self.inp_length = inp_length
        self.channels = channels
        self.label_size = label_size
        self.cnn_layers = cnn_layers
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.cnn_dropout = cnn_dropout
        self.lstm_dropout = lstm_dropout
        self.dense_dropout = dense_dropout
        self.cnn_filters = cnn_filters
        self.activation = activation
        self.padding = padding
        self.lstm_neurons = lstm_neurons
        self.dense_neurons = dense_neurons
        self.cnn_filter_size = cnn_filter_size
        self.pool_size = pool_size
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.monitor = monitor
        self.patience = patience



    def __call__(self, inp_p, inp_s, dists):

        # P encoding branch
        x_p = inp_p
        for i in range(self.cnn_layers):
            x_p = Conv1D(self.cnn_filters[i], 
                         self.cnn_filter_size[i], 
                         padding=self.padding,
                         activation=self.activation
                         )(x_p)
            x_p = Dropout(self.cnn_dropout)(x_p, training=True)
            x_p = MaxPooling1D(self.pool_size, padding=self.padding)(x_p)
        

        for i in range(self.lstm_layers):
            if i == self.lstm_layers - 1:
                x_p = LSTM(self.lstm_neurons[i],  
                                    dropout=self.lstm_dropout, 
                                    recurrent_dropout=self.lstm_dropout, 
                                    return_sequences=False,
                                    activation=self.activation
                                    )(x_p)
            else:
                x_p = Bidirectional(LSTM(self.lstm_neurons[i], 
                                    dropout=self.lstm_dropout, 
                                    recurrent_dropout=self.lstm_dropout, 
                                    return_sequences=True,
                                    activation=self.activation
                                    ))(x_p)


        # S encoding branch
        x_s = inp_s
        for i in range(self.cnn_layers):
            x_s = Conv1D(self.cnn_filters[i], 
                        self.cnn_filter_size[i], 
                        padding=self.padding,
                        activation=self.activation,
                        )(x_s)
            x_s = Dropout(self.cnn_dropout)(x_s, training=True)
            x_s = MaxPooling1D(self.pool_size, padding=self.padding)(x_s)
        

        for i in range(self.lstm_layers):
            if i == self.lstm_layers - 1:
                x_s = LSTM(self.lstm_neurons[i],
                                    dropout=self.lstm_dropout, 
                                    recurrent_dropout=self.lstm_dropout, 
                                    return_sequences=False,
                                    activation=self.activation
                                    )(x_s)
            else:
                x_s = Bidirectional(LSTM(self.lstm_neurons[i], 
                                    dropout=self.lstm_dropout, 
                                    recurrent_dropout=self.lstm_dropout, 
                                    return_sequences=True,
                                    activation=self.activation
                                    ))(x_s)
        

        # x = Lambda(lambda tensors: concatenate(tensors, axis=-1))([Flatten()(x_p), Flatten()(x_s), dists])
        x = concatenate([Flatten()(x_p), Flatten()(x_s), dists])


        for i in range(self.dense_layers):
            if i < self.dense_layers - 1:
                x = Dense(self.dense_neurons[i], activation=self.activation)(x)
                x = Dropout(self.dense_dropout)(x)
            else:
                x = Dense(self.label_size)(x)

        o = Activation('linear', name='output')(x)


        model = Model(inputs=[inp_p, inp_s, dists], outputs=o)
        opt_engine = self.optimizer(learning_rate = self.learning_rate)
        model.compile(optimizer=opt_engine, loss=self.loss)
        model.summary()


        return model




class BuildModel_Enco_Deco():
    
    def __init__(self,
                 inp_length = None,
                 channels = 3,
                 label_size = None,
                 cnn_layers = None,
                 lstm_layers = None,
                 cnn_dropout = None,
                 lstm_dropout =None,
                 cnn_filters = None,
                 padding = None,
                 lstm_neurons = None,
                 cnn_filter_size = None,
                 activation = None,
                 optimizer = None,
                 loss = None,
                 epochs = None,
                 batch_size = None,
                 learning_rate = None,
                 monitor = None,
                 patience = None
                 ):
        
        self.inp_length = inp_length
        self.channels = channels
        self.label_size = label_size
        self.cnn_layers = cnn_layers
        self.lstm_layers = lstm_layers
        self.cnn_dropout = cnn_dropout
        self.lstm_dropout = lstm_dropout
        self.cnn_filters = cnn_filters
        self.padding = padding
        self.lstm_neurons = lstm_neurons
        self.cnn_filter_size = cnn_filter_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.monitor = monitor
        self.patience = patience


    def __call__(self, inp):

        x = inp
        for i in range(self.cnn_layers):
            x = BatchNormalization()(x)
            x = Conv1D(self.cnn_filters[i], 
                       self.cnn_filter_size[i], 
                       padding=self.padding, 
                       kernel_regularizer = keras.regularizers.l1(1e-4),
                       bias_regularizer = keras.regularizers.l1(1e-4),
                       )(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            x = Dropout(self.cnn_dropout)(x, training=True)
            x = MaxPooling1D(2, padding=self.padding)(x)


        for i in range(self.lstm_layers):
            if i == self.lstm_layers - 1:
                x = Bidirectional(GRU(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                        recurrent_dropout=self.lstm_dropout, return_sequences=True))(x)
                x = Conv1D(self.lstm_neurons[i], 1, padding=self.padding)(x)
                x = BatchNormalization()(x)      
            else:          
                x = Bidirectional(LSTM(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                        recurrent_dropout=self.lstm_dropout, return_sequences=True))(x)
                x = Conv1D(self.lstm_neurons[i], 1, padding=self.padding)(x)
                x = BatchNormalization()(x)

        
        rev_filt = [f for f in reversed(self.cnn_filters)]
        rev_size = [s for s in reversed(self.cnn_filter_size)]
        for i in range(self.cnn_layers):
            x = UpSampling1D(2)(x)
            if i == self.cnn_layers//2: x = Cropping1D(cropping=(2,2))(x)
            x = Conv1D(rev_filt[i], 
                       rev_size[i], 
                       padding=self.padding, 
                       activation = self.activation,
                       kernel_regularizer = keras.regularizers.l1(1e-4),
                       bias_regularizer = keras.regularizers.l1(1e-4),
                       )(x)
            x = Dropout(self.cnn_dropout)(x, training=True)

        mag = Conv1D(1, 11, padding=self.padding, activation='sigmoid', name='output_layer')(x)

        model = Model(inputs=inp, outputs=mag)
        opt_egine = self.optimizer(learning_rate = self.learning_rate)
        model.compile(optimizer=opt_egine, loss=self.loss)
        model.summary()

        return model
