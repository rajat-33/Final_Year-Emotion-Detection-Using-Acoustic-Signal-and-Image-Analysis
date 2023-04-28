import pickle
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

paths = []
labels = []
for dirname, _, filenames in os.walk('./data/TESS Toronto emotional speech set data'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break
print('Dataset is Loaded')

df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()
df['label'].value_counts()
# sns.countplot(df['label'])


def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


# emotion = 'fear'
# path = np.array(df['speech'][df['label'] == emotion])[0]
# data, sampling_rate = librosa.load(path)
# waveplot(data, sampling_rate, emotion)
# spectogram(data, sampling_rate, emotion)
# Audio(path)


def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# print(extract_mfcc(df['speech'][0]))
X_mfcc = []
X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))

X = [x for x in X_mfcc]
X = np.array(X)
print(X.shape)

enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()

x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=0, shuffle=True)

# ------------ preparing test data

paths2 = []
labels2 = []

Savee = "./data/ALL/"
savee_directory_list = os.listdir(Savee)
for file in savee_directory_list:
    paths2.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele == 'a':
        labels2.append('angry')
    elif ele == 'd':
        labels2.append('disgust')
    elif ele == 'f':
        labels2.append('fear')
    elif ele == 'h':
        labels2.append('happy')
    elif ele == 'n':
        labels2.append('neutral')
    elif ele == 'sa':
        labels2.append('sad')
    else:
        labels2.append('surprise')

print('test dataset is loaded')

df2 = pd.DataFrame()
df2['speech'] = paths2
df2['label'] = labels2
df2.head()
df2['label'].value_counts()

test_mfcc = df2['speech'].apply(lambda x: extract_mfcc(x))

test_data = [x for x in test_mfcc]
test_datax = np.array(test_data)
enc2 = OneHotEncoder()
y = enc2.fit_transform(df2[['label']])
test_datay = y.toarray()
# ------------ done test data loading

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_data=(
    test_datax, test_datay), epochs=50, batch_size=64)

model_name = "modelForValidation.pkl"
with open(model_name, 'wb') as file:
    pickle.dump(model, file)
