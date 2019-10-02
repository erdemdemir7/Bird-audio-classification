import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras import models
from keras import layers

from functions import playAudio
from functions import plotAudio
from functions import initiate_birds
from functions import initiate_libr

#Initials
path = "/Users/erdemdemir/Desktop/bird-dir/bird-types.txt"
path_to_import = "/Users/erdemdemir/Desktop/bird-sounds/"
bird_names = open(path, "r")

#Birds List
birds = initiate_birds()

#Birds and their songs Dictionary
libr = initiate_libr(birds)

'''
#To print the dictionary
for keys in libr.keys():
    print("keys:{}:{} {}\n".format(keys, len(libr[keys]),libr[keys]))
'''
#Play ond plot bird songs counter # of times
paths = []
counter = 10

for t in libr.keys():
    for x in libr[t]:
        if counter == 0:
            break
        pth = path_to_import + t +"/" + x
        paths.append(pth)
        counter -= 1


# generating a dataset
header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for g in birds:
    for filename in libr[g]:
        songname = "{}{}/{}".format(path_to_import, g, filename)
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename}~~ {np.mean(chroma_stft)}~~ {np.mean(rms)}~~ {np.mean(spec_cent)}~~ {np.mean(spec_bw)}~~ {np.mean(rolloff)}~~ {np.mean(zcr)}~~'
        for e in mfcc:
            to_append += f' {np.mean(e)}~~'
        to_append += f' {g}~~'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split('~~'))

# reading dataset from csv

data.head()


'''
#Neural Network Process


# Dropping unneccesary columns
data = data.drop(['filename'], axis=1)
data.head()

bird_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(bird_list)
print(y)

# normalizing
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

# spliting of dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# creating a model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)

# calculate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc: ', test_acc)

# predictions
predictions = model.predict(X_test)
np.argmax(predictions[0])
'''