import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import librosa
import scipy
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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

from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

SAMPLE_RATE = 44100

#Initials
path = os.getcwd() + "/"
song = os.getcwd() + "/bird-sounds/"
bird_path = path + "bird-dir/bird-types.txt"
bird_names = open(bird_path, "r")

#Birds List
birds = initiate_birds()

#Birds and their songs Dictionary
libr = initiate_libr(birds)

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
print('done loading train mfcc')
test_data = test_data['song'].apply(get_mfcc, path=test['song'][0:])
print('done loading test mfcc')

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



def proba2labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids

#fitting random forest on the dataset
rfc = RandomForestClassifier(n_estimators = 150)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, shuffle = True)

#fitting on the entire data
model = rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
str_preds, _ = proba2labels(rfc.predict_proba(test_data.drop('bird', axis = 1).values), i2c, k=3) # For submission.csv

cm = confusion_matrix(y_test, predictions)
tmp = ConfusionMatrix(actual_vector=y_test, predict_vector=predictions)

cmd = pd.DataFrame(cm, index=tmp.classes, columns=tmp.classes)

# Plotting results
plt.figure(figsize=(7,7))
sns.heatmap(cmd, annot=True, cmap='RdPu')
plt.title('RFC Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, predictions)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("\n\nlabel precision recall")
for label in range(len(cm)):
    print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")

print("\nprecision total:", precision_macro_average(cm))
print("recall total:", recall_macro_average(cm))

print(f'\nAccuracy: {accuracy(cm)}')

'''
# Plotting results
plt.scatter(y_test, predictions)
plt.xlabel('True_Values')
plt.ylabel('Predictions')
plt.title('Model Random_Forest:')
plt.show()
'''

#checking the accuracy of the model
print(f'\nModel Random_Forest:\n')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Accuracy of Model:', rfc.score(X_test, y_test))



#Model
#print(f'\n\n {model}')

# Prepare submission
subm = pd.DataFrame()
subm['song'] = test['song']
subm['bird'] = str_preds
subm.to_csv('submission.csv', index=False)


