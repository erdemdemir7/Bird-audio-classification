import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
import os
import csv
import librosa
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from functions import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from pycm import *

from functions import initiate_birds
from functions import initiate_libr

from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)


#Initials
path = os.getcwd() + "/"
song = os.getcwd() + "/bird-sounds/"
bird_path = path + "bird-dir/bird-types.txt"
bird_names = open(bird_path, "r")

SAMPLE_RATE = 44100

#Birds List
birds = initiate_birds()

#Birds and their songs Dictionary
libr = initiate_libr(birds)

#Read Dataset
train = pd.read_csv(f'{path}train.csv')
test = pd.read_csv(f'{path}test.csv')

train = pd.DataFrame(train)
test = pd.DataFrame(test)


def get_mfcc(name, path):
    b, _ = librosa.core.load(name, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    gmm = librosa.feature.mfcc(b, sr = SAMPLE_RATE, n_mfcc=20)
    return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1))))


train_data = pd.DataFrame()

train_data['song'] = train['song']
test_data = pd.DataFrame()
test_data['song'] = test['song']

train_data = train_data['song'].apply(get_mfcc, path=train['song'][0:])
test_data = test_data['song'].apply(get_mfcc, path=test['song'][0:])

train_data['bird'] = train['bird']
test_data['bird'] = np.zeros((len(test['song'])))


X = train_data.drop('bird', axis=1)
feature_names = list(X.columns)
X = X.values
num_class = len(birds)
c2i = {}
i2c = {}
for i, c in enumerate(birds):
    c2i[c] = i
    i2c[i] = c
y = np.array([c2i[x] for x in train_data['bird'][0:]])

#Define scores list
scores = []

#fitting random forest on the dataset
rfc = RandomForestClassifier(n_estimators = 150)

#5-Fold-Cross-Validation
cv = KFold(n_splits=5, random_state=None, shuffle=True)

counter = 0 # To state the model

for train_index, test_index in cv.split(X):
    counter += 1
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    rfc.fit(X_train, y_train)
    scores.append(rfc.score(X_test, y_test))
    y_pred = rfc.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tmp = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred)

    cmd = pd.DataFrame(cm, index=tmp.classes, columns=tmp.classes)

    # Printing metrics
    print(f'\nRFC\nModel {counter}:\n')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Accuracy of Model:', rfc.score(X_test, y_test))

    # Plotting results
    plt.figure(figsize=(7, 7))
    sns.heatmap(cmd, annot=True, cmap='RdPu')
    txt1 = 'RFC Linear Kernel '
    txt2 = '\nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred))
    txt3 = f' Model {counter}:'
    txt = txt1 + txt3 + txt2
    plt.title(txt)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Printing Confusion matrix details
    print("\n\nlabel precision recall")
    for label in range(len(cm)):
        print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")

    print("\nprecision total:", precision_macro_average(cm))
    print("recall total:", recall_macro_average(cm))

    print(f'\nAccuracy: {accuracy(cm)}')
    print('\n--------------------------------------------\n\n')


    '''
    # Plotting results
    plt.scatter(y_test, y_pred)
    plt.xlabel('True_Values')
    plt.ylabel('Predictions')
    plt.title(f'Model {counter}:')
    plt.show()
    '''

# Accuracy Results
print(f'\nScores: {scores}\n')

# Mean of Accuracy Results
disp_mean = "{:.2f}".format(100 * np.mean(scores))
print(f'Accuracy of 5-Fold-Cross-Validation \nMean: {disp_mean}%\n')

# Std. Dev. of Accuracy Results
disp_std = "{:.2f}".format(np.std(scores))
print(f'Standard Deviation: {disp_std}\n')




'''
# Sklearn cross val.
c_scores = cross_val_score(rfc, X, y, cv=5)
print(f'\nCross-validated scores: {c_scores}')

# Make cross validated predictions
predictions = cross_val_predict(rfc, X, y, cv=5)
plt.scatter(y, predictions)
plt.xlabel('True_Values')
plt.ylabel('Predictions')
plt.title('Model Pred:')
plt.show()
'''