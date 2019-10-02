import pyaudio
import wave
import sys
import time
import struct
import numpy as np
import matplotlib.pyplot as plt
import functions
import os


# Helper functions that are used to implement other methods

#Read the audio file and play
def playAudio(file_name):
    # Open the sound file
    wf = wave.open(file_name, 'r')

    # Create an interface to PortAudio
    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        return (data, pyaudio.paContinue)

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    wf.close()

    p.terminate()

#Read the audio file and plot
def plotAudio(file_name):
    spf = wave.open(file_name, 'r')
    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = spf.getframerate()

    Time = np.linspace(0, len(signal) / fs, num=len(signal))

    plt.figure(1)
    plt.title(file_name.split('/')[len(file_name.split('/'))-1])
    plt.plot(Time, signal)
    plt.xlabel('sec')
    plt.ylabel('Hz')
    plt.show()

#Audio-processing helper functions
pth = os.getcwd() + "/"
path = pth + "bird-dir/bird-types.txt"
bird_names = open(path, "r")


def initiate_birds():
    birds = []
    for each in bird_names:
        birds.append(each.replace('\n',''))
    birds.sort()
    return birds

def initiate_libr(birds):
    libr = dict()
    for x in birds:
        libr.update({x: list()})
    initiate_libr_helper(libr)
    return libr

def initiate_libr_helper(libr):
    bird_names = open(path, "r")
    for n in bird_names:
        path_t = pth + "bird-dir/" + n.replace('\n','') + ".txt"
        name_is = open(path_t, "r")
        for t in name_is:
            libr.update(name=libr[n.replace('\n','')].append(t.replace('\n','')))
        libr.pop('name')
    return libr


# Functions for the confusion Matrix
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements