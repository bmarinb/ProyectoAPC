#!/usr/bin/env python
# coding: utf-8

# In[4]:


import scipy.io.wavfile
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io
import pywt
import librosa

#  https://stackoverflow.com/questions/39032325/python-high-pass-filter

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


def homomorphic_envelope(x, fs=1000, f_LPF=8, order=3):
    """
    Computes the homomorphic envelope of x

    Args:
        x : array
        fs : float
            Sampling frequency. Defaults to 1000 Hz
        f_LPF : float
            Lowpass frequency, has to be f_LPF < fs/2. Defaults to 8 Hz
    Returns:
        time : numpy array
    """
    b, a = scipy.signal.butter(order, 2 * f_LPF / fs, 'low')
    he = np.exp(scipy.signal.filtfilt(b, a, np.log(np.abs(scipy.signal.hilbert(x)))))
    return he 

def hilbert_transform(x):
    """
    Computes modulus of the complex valued
    hilbert transform of x
    """
    return np.abs(scipy.signal.hilbert(x))

def wavelet(signal):
    L = pywt.wavedec(signal, "rbio3.9", level=3)
    print(signal.shape)
    for i in L:
        print(i.shape)
    return L


def load_mat(varname):
    mat = scipy.io.loadmat(varname+'.mat')
    return mat[varname]


# In[6]:


wavs = load_mat("example_audio_data")
patient_numbers = load_mat("patient_number")
labels = load_mat("labels")
n_train = (1,100)
n_val = (101,111)
n_test = (111,136)
numbers = np.array([int(patient_numbers[0][i][0][0])for i in range(patient_numbers.shape[1])])

train_delim = min(np.argwhere(numbers==n_val[0]))[0] # Primer miembro de VAL
val_delim =  min(np.argwhere(numbers==n_test[0]))[0] # Primer miembro de TEST


# In[7]:



def split_set(arr, train_delim, val_delim):
    if arr.shape[0]>1:
        train_arr = arr[:train_delim]
        val_arr = arr[train_delim:val_delim]
        test_arr = arr[val_delim:]
    else:
        train_arr = arr[0,:train_delim]
        val_arr = arr[0,train_delim:val_delim]
        test_arr = arr[0,val_delim:]
    return train_arr, val_arr, test_arr

train_wavs, val_wavs, test_wavs = split_set(wavs, train_delim, val_delim)
train_labels, val_labels, test_labels = split_set(labels, train_delim, val_delim)


# In[144]:


def zero_pad(arr, size):
    new_arr = np.zeros(size)
    new_arr[:arr.shape[0]] = arr
    return new_arr

def count_nan(arr):
    return np.sum(np.isnan(arr).flatten())
def extract_features(signal, sr, n_feats=100, red=2):
    
    length = signal.shape[0]

    filtered_signal = butter_bandpass_filter(signal, 25, 400, sr, order=4)
    signal = np.array(filtered_signal)
    
    
    delta = 20
    largo_segmento = 80
    overlap = largo_segmento-delta

    f, t, Zxx = scipy.signal.stft(signal, sr, 'hamming', nperseg=largo_segmento, noverlap=overlap)
    h_feat = hilbert_transform(signal)
    
    #homom = homomorphic_envelope(signal)
    #homom = np.resize(homom, (delta, int(length/delta)))
    h_feat = np.resize(h_feat, (delta, int(length/delta)))
    if Zxx.shape[1]> n_feats:
        Zxx = Zxx[:,:n_feats]
        
    
    #print(Zxx.shape, homom.shape, h_feat.shape)
    Z = np.concatenate([Zxx, h_feat], axis=0)
    #print(Z.shape)
    L = []
    if False:
        # Wavelet y MFCC quedan afuera por no contar con código eficiente
        mfcc = librosa.feature.mfcc(signal, sr=sr)
        wt = wavelet(filtered_signal)
    return Z


# In[145]:


max_len = 0
max_label_len = 0

for i in range(wavs.shape[1]):
    wav_len = wavs[0,i].shape[0]
    if wav_len > max_len:
        max_len = wav_len
        max_label_len = labels[i,0].shape[0]
        
red = 2
max_len = int(max_len/red) + 20 
max_label_len = int(max_label_len/red) + 1
num_feats = 61


# In[150]:


def generate_X_y(wavs, labels, num_feats, max_label_len, red):
    X = np.zeros((wavs.shape[0], num_feats, max_label_len))
    y = np.zeros((wavs.shape[0], max_label_len), dtype=np.int8)
    sr = 2000
    new_sr = sr/red
    for i in range(0,wavs.shape[0]):
        if i%100==0:
            print(100*i/wavs.shape[0], "%")
        label = labels[i][0]
        label = label[range(0,label.shape[0], 2)]
        label = zero_pad(label, (max_label_len,1))    
        y[i, :] = np.squeeze(label)
        signal = wavs[i].flatten()
        resampled_length = int(signal.shape[0]*new_sr/sr)
        signal = scipy.signal.resample(signal, resampled_length)
        signal = zero_pad(signal, max_len)    
        n_feats = label.shape[0]
        feats = extract_features(signal, new_sr,n_feats = n_feats, red=red)
        X[i, :, :] = feats
    X = np.transpose(X,axes=[0,2,1])
    return X, y


# In[151]:


X_train, y_train= generate_X_y(train_wavs, train_labels, num_feats, max_label_len, red)
X_val, y_val= generate_X_y(val_wavs, val_labels, num_feats, max_label_len, red)
X_test, y_test = generate_X_y(test_wavs, test_labels, num_feats, max_label_len, red)
# La razón entre labels y largo del audio es 1:20. Esto significa que puedo usar un algoritmo que extraiga características


# In[173]:


from sklearn.preprocessing import StandardScaler

X_scale = np.reshape(X_train, (X_train.shape[0]* X_train.shape[1],61))

scaler = StandardScaler()
scaler.fit(X_scale)
print(X_train.shape)

def transform(X):
    for i in range(X.shape[1]):
        X[:,i,:] = scaler.transform(X[:,i, :])
    return X

X_train2 = transform(X_train)
X_val2 = transform(X_val)
X_test2 = transform(X_test)


# In[ ]:


def one_hot_y(y):
    yhot = np.zeros((y.shape[0], y.shape[1], 5))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            yhot[i,j,y[i,j]] = 1
    return yhot

yhot_train = one_hot_y(y_train)
yhot_val = one_hot_y(y_val)
yhot_test = one_hot_y(y_test)


# In[ ]:


from keras.layers import Dense,CuDNNGRU, Bidirectional, GRU, Activation, Flatten
from keras.models import Sequential
import tensorflow as tf
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam


def create_model():
    model = Sequential()


    model.add(GRU(10, return_sequences=True, input_shape=( X_train.shape[1], X_train.shape[2])))
    model.add(GRU(10, return_sequences=True, input_shape=( X_train.shape[1], X_train.shape[2])))

    model.add(Dense(5))
    model.add(Activation('softmax'))
    optimizer = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()
    checkpoint = ModelCheckpoint("Model.hdf5", save_best_only=True)
    stopping = EarlyStopping(min_delta=0.0001, patience=10, restore_best_weights=True)
    callbacks = [checkpoint, stopping]
    history = model.fit(X_train, yhot_train, epochs=20, batch_size=256, validation_data=(X_val, yhot_val), callbacks=callbacks)
    return model

model = create_model()


# In[176]:


model.predict(X_test)


# In[ ]:




