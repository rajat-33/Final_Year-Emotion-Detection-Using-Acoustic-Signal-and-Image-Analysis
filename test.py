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
import warnings
warnings.filterwarnings('ignore')


# def extract_mfcc(filename):
#     y, sr = librosa.load(filename, duration=3, offset=0.5)
#     mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
#     return mfcc


# paths = []
# labels = []
# for dirname, _, filenames in os.walk('./data/TESS Toronto emotional speech set data'):
#     for filename in filenames:
#         paths.append(os.path.join(dirname, filename))
#         label = filename.split('_')[-1]
#         label = label.split('.')[0]
#         labels.append(label.lower())
#     if len(paths) == 2800:
#         break
# print('Dataset is Loaded')

# df = pd.DataFrame()
# df['speech'] = paths
# df['label'] = labels
# df.head()
# df['label'].value_counts()

# X_mfcc = []
# X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))

# X = [x for x in X_mfcc]
# X = np.array(X)
# print(X.shape)


# # ------------ preparing test data

# paths2 = []
# labels2 = []

# Savee = "./data/ALL/"
# savee_directory_list = os.listdir(Savee)
# for file in savee_directory_list:
#     paths2.append(Savee + file)
#     part = file.split('_')[1]
#     ele = part[:-6]
#     if ele == 'a':
#         labels2.append('angry')
#     elif ele == 'd':
#         labels2.append('disgust')
#     elif ele == 'f':
#         labels2.append('fear')
#     elif ele == 'h':
#         labels2.append('happy')
#     elif ele == 'n':
#         labels2.append('neutral')
#     elif ele == 'sa':
#         labels2.append('sad')
#     else:
#         labels2.append('surprise')

# print('test dataset is loaded')

# df2 = pd.DataFrame()
# df2['speech'] = paths2
# df2['label'] = labels2
# df2.head()
# df2['label'].value_counts()

# test_mfcc = df2['speech'].apply(lambda x: extract_mfcc(x))

# test_data = [x for x in test_mfcc]
# test_datax = np.array(test_data)
# enc2 = OneHotEncoder()
# y = enc2.fit_transform(df2[['label']])
# test_datay = y.toarray()


Ravdess = "./data/audio_speech_actors_01-24/"
Crema = "./data/AudioWAV/"
Tess = "./data/TESS Toronto emotional speech set data/"
Savee = "./data/ALL/"

ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.Emotions.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                            5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)
Ravdess_df.head()

crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part = file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df.head()


tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[-1]  # [2] should be there
        if part == 'ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
Tess_df.head()

savee_directory_list = os.listdir(Savee)

file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele == 'a':
        file_emotion.append('angry')
    elif ele == 'd':
        file_emotion.append('disgust')
    elif ele == 'f':
        file_emotion.append('fear')
    elif ele == 'h':
        file_emotion.append('happy')
    elif ele == 'n':
        file_emotion.append('neutral')
    elif ele == 'sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)
Savee_df.head()


# creating Dataframe using all the 4 dataframes we created so far.
data_path = pd.concat([Ravdess_df, Crema_df, Savee_df], axis=0)
data_path.to_csv("data_path2.csv", index=False)
data_path.head()


def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def create_spectrogram(data, sr, e):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.show()
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


# def stretch(data, rate=0.8):
#     return


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


# def pitch(data, sampling_rate, pitch_factor=0.7):
#     return


# taking any example and checking for techniques.
path = np.array(data_path.Path)[1]
data, sample_rate = librosa.load(path)


plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=data, sr=sample_rate)
# plt.show() to show the plot
Audio(path)


x = noise(data)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
# plt.show()
Audio(x, rate=sample_rate)

x = librosa.effects.time_stretch(data, rate=0.8)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
Audio(x, rate=sample_rate)

x = shift(data)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
Audio(x, rate=sample_rate)


# pitch is not included
x = librosa.effects.pitch_shift(
    data, sr=sample_rate, n_steps=6)  # included with changes
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
# plt.show()
Audio(x, rate=sample_rate)


# feature extraction Audio->numbers

def extract_features(data):
    # # ZCR
    # result = np.array([])
    # zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    # result = np.hstack((result, zcr))  # stacking horizontally
    # print(zcr.shape)

    # # Chroma_stft
    # stft = np.abs(librosa.stft(data))
    # chroma_stft = np.mean(librosa.feature.chroma_stft(
    #     S=stft, sr=sample_rate).T, axis=0)
    # result = np.hstack((result, chroma_stft))  # stacking horizontally
    # print(chroma_stft.shape)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    # result = np.hstack((result, mfcc))  # stacking horizontally
    return mfcc
    # print(mfcc.shape)

    # # Root Mean Square Value
    # rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    # result = np.hstack((result, rms))  # stacking horizontally
    # print(rms.shape)

    # # MelSpectogram
    # mel = np.mean(librosa.feature.melspectrogram(
    #     y=data, sr=sample_rate).T, axis=0)
    # result = np.hstack((result, mel))  # stacking horizontally
    # print(mel.shape)

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    # print(res1)
    # print("this is shape: ", res1.shape)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = librosa.effects.time_stretch(data, rate=0.8)
    data_stretch_pitch = librosa.effects.pitch_shift(
        data, sr=sample_rate, n_steps=6)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result


# X, Y = [], []
# for path, emotion in zip(data_path.Path, data_path.Emotions):
#     print(len(X))
#     if len(X) > 9000:
#         break
#     feature = get_features(path)
#     for ele in feature:
#         X.append(ele)
#         # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
#         Y.append(emotion)

# len(X), len(Y), data_path.Path.shape

# print(len(X[0]))

Features = pd.read_csv("./features2.csv")
Features.to_csv('features2.csv', index=False)
Features.head()

X = Features.iloc[:, :-1].values
Y = Features['labels'].values
# print(Y)

# print(X[0])
# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
# ------------ done test data loading

enc = OneHotEncoder()
y = enc.fit_transform(data_path[['Emotions']])
y = y.toarray()

model_name = "LSTMmodel.pkl"
with open(model_name, 'rb') as file:
    model = pickle.load(file)


data, sample_rate = librosa.load("./data/test/YAF_beg_ps.wav")
input = extract_features(data)
testFile = np.array([input])
testFile = np.expand_dims(testFile, -1)
print(testFile.shape)
yPred = model.predict(testFile)
print(yPred)
predictedEmotion = enc.inverse_transform(yPred)
print(predictedEmotion)
here = enc.transform([['happy'], ['angry'], ['sad'], ['fear'], [
    'disgust'], ['neutral'], ['calm'], ['surprise']]).toarray()
print(here)
# print("Accuracy of our model on test data : ",
#       model.evaluate(test_datax, test_datay)[1]*100, "%")
