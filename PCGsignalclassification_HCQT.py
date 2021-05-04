# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:15:43 2021

@author: shamik.tiwari
"""
import glob
import os
import librosa as lib
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import IPython.display as ipd
import shutil
import soundfile
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.utils import shuffle

def load_files(path):
    files = [file for file in glob.glob(path)]
    return files

pcgdataset = []
for folder in ["E://set_a/**","E://set_b/**"]:
    for filename in glob.iglob(folder):
        if os.path.exists(filename):
            label = os.path.basename(filename).split("_")[0]
            # skip audio smaller than 4 secs
            if librosa.get_duration(filename=filename)>=3:
              if label not in ["Aunlabelledtest", "Bunlabelledtest"]:
                pcgdataset.append({
                        "filename": filename,
                        "label": label
                    })
pcgdataset = pd.DataFrame(pcgdataset)
pcgdataset = shuffle(pcgdataset, random_state=42)
plt.figure(figsize=(12,6))
pcgdataset.label.value_counts().plot(kind='bar', title="Dataset distribution")
plt.show()

unique_labels = pcgdataset.label.unique()


#Changing only the speed of an audio signal with different rates and saving it
def speedchange(speed_rate,src_path,dst_path):
    files = load_files(src_path + "//**")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for file in tqdm(files):
        label = os.path.basename(file).split(".")[0]
        y, sr = lib.load(file) 
        updated_y = lib.effects.time_stretch(y, rate=speed_rate)
        soundfile.write(dst_path +"//" + label + "_" + str(speed_rate) + ".wav",updated_y, sr)
        
#Changing only the pitch of an audio signal with different steps and saving it
def pitchchange(step,src_path,dst_path):
    files = load_files(src_path + "//**")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for file in tqdm(files):
        label = os.path.basename(file).split(".")[0]
        y, sr = lib.load(file)
        updated_y = lib.effects.pitch_shift(y, sr, n_steps=step)
        soundfile.write(dst_path +"//" + label + "_" + str(step) + ".wav",updated_y, sr)

def pcgaugmentation(src_path,dst_path):
    speed_rates = [1.02,0.9,1.05,0.8]
    for speed_rate in speed_rates:
        speedchange(speed_rate,src_path,dst_path)
        
    steps = [1.5,-1.5,2.0,-2.0]
    for step in steps:
        pitchchange(step,src_path,dst_path)
        
    files = load_files(src_path + "//**")
    for f in files:
        shutil.copy(f,dst_path)

if os.path.exists("E:\\set_a"):
    if len(load_files("E:\\set_a\\**")) == 4175:
        print("Sound Augumentation Already Done and Saved")
    else:
        shutil.rmtree("E:\\set_a")
        pcgaugmentation("E:\\set_b","E:\\set_a")
else:
    pcgaugmentation("E:\\set_b","E:\\set_a")


pcgdata = []
for folder in ["E://set_a/**"]:
    for filename in glob.iglob(folder):
        if os.path.exists(filename):
            label = os.path.basename(filename).split("_")[0]
            # skip audio smaller than 4 secs
            if librosa.get_duration(filename=filename)>=3:
              if label not in ["Aunlabelledtest", "Bunlabelledtest"]:
                pcgdata.append({
                        "filename": filename,
                        "label": label
                    })
pcgdata = pd.DataFrame(pcgdata)
pcgdata = shuffle(pcgdata, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=37)
for train_idx, test_idx in split.split(pcgdata, pcgdata.label):
    train = pcgdata.loc[train_idx]
    test = pcgdata.loc[test_idx]



def feature(file_path):
    y, sr = lib.load(file_path,duration=3)
    fmin = librosa.midi_to_hz(36)
    hop_length = 512
    #C=librosa.cqt(y, sr=sr, hop_length=hop_length)
    C=librosa.hybrid_cqt(y, sr=sr, hop_length=hop_length)
    #C = librosa.vqt(y, sr=sr, hop_length=hop_length)
    return C
x_train = np.asarray([feature(train.filename.iloc[i]) for i in (range(len(train)))])
x_test = np.asarray([feature(test.filename.iloc[i]) for i in (range(len(test)))])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


encode = LabelEncoder()
y_train = encode.fit_transform(train.label)
y_test = encode.fit_transform(test.label)
y_train = to_categorical(y_train,num_classes=5)
y_test = to_categorical(y_test,num_classes=5)


from keras.models import Model,Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam


def CNN_model(n_width,n_height,n_channels,n_dropout,n_classes):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1, 1),input_shape=(n_width,n_height,n_channels), activation ='relu'))
    cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    cnn_model.add(Dropout(rate=n_dropout))    
    cnn_model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'valid', activation ='relu'))
    cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    cnn_model.add(Dropout(rate=0.2))
    cnn_model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'valid', activation ='relu'))
    cnn_model.add(Conv2D(filters=32, kernel_size=(5,5), padding = 'valid', activation ='relu'))
    cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    cnn_model.add(Dropout(rate=0.2))
    cnn_model.add(Flatten())
    cnn_model.add(Dropout(rate=n_dropout))

    cnn_model.add(Dense(64, activation ='relu'))
    cnn_model.add(Dropout(rate=n_dropout))

    cnn_model.add(Dense(n_classes, activation ='softmax'))
    
    return cnn_model
cnn_model = CNN_model(x_train.shape[1],x_train.shape[2],x_train.shape[3],0.5,len(encode.classes_))



optimizer = Adam(lr=0.0001)
cnn_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
cnn_model.summary()

epochs = 300
batch_size = 128
file = 'cnn_heartbeat_classifier.hdf5'
path = os.path.join(file)

checkpoints = ModelCheckpoint(filepath=path,save_best_only=True,verbose=1)

cnn_history = cnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                        callbacks=[checkpoints],verbose=1)


preds = cnn_model.predict(x_test)
y_actual = []
y_pred = []

labels = encode.classes_
for idx, pred in enumerate(preds): 
    y_actual.append(labels[np.argmax(y_test[idx])])
    y_pred.append(labels[np.argmax(pred)])

print(classification_report(y_pred, y_actual))

history = cnn_history.history

#Plotting epoch vs Training and Testing accuracy Graph
plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 16})
plt.plot(history['accuracy'], 'b+', linewidth=2, markersize=8)
plt.plot(history['val_accuracy'], 'g+', linewidth=2, markersize=8)
plt.title('Model accuracy using HCQT transform features')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy','Testing Accuracy'],loc='upper left') 
plt.grid()
plt.show()

#Plotting epoch vs Training and Testing loss Graph
plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 16})
plt.plot(history['loss'], 'b+', linewidth=2, markersize=4)
plt.plot(history['val_loss'], 'r+', linewidth=2, markersize=4)
plt.title('Model loss using HCQT transform features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss','Testing Loss'],loc='upper right')
plt.grid()
plt.show()

#Creating a confusion matrix
mat = confusion_matrix(y_actual,y_pred)
fig, ax = plt.subplots(figsize=(10,10))
plt.rcParams.update({'font.size': 20})
plot_confusion_matrix(conf_mat=mat,cmap = "OrRd",colorbar=True, show_absolute=False,show_normed=True,figsize=(10,10),class_names=encode.classes_)
plt.show()


from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle
n_classes=5
lw=2
y_score = preds
target=y_test
fpr = dict()
tpr = dict()
roc_auc = dict()
fig, ax = plt.subplots(figsize=(20,20))
plt.rcParams.update({'font.size': 12})
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(target[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=3)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['b', 'g', 'r', 'm',  'k'])
classes = ['artifact', 'extrahls', 'extrastole','murmur','normal']
for (i, j, color) in zip(range(n_classes),classes, colors):
       plt.plot(fpr[i], tpr[i], color=color, lw=lw,label=j+'(area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

    
plt.plot([0, 1], [0, 1], 'k+', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for for heart beat classification using HCQT transform features')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt
y, sr = librosa.load("E:\\set_b\\normal__106_1306776721273_B1.wav")
C = np.abs(librosa.cqt(y, sr=sr))
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                               sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")