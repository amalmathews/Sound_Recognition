 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join
import time
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pandas as pd
import os
import warnings
import librosa
import numpy as np
import tensorflow as tf

def warn(*args, **kwargs):
    pass





fulldatasetpath = './Data/audio/'
metadata = pd.read_csv('./Data/metadata/UrbanSound8K.csv')
features = []
features2 = []

for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath), 'fold' + str(row["fold"]) + '/',
                             str(row["slice_file_name"]))
    class_label = row["class"]
    classid = row["classID"]
    features2.append([class_label])
# print(features2)
features2df = pd.DataFrame(features2, columns=['class_label'])
y2 = np.array(features2df.class_label.tolist())

le = LabelEncoder()
yy2 = to_categorical(le.fit_transform(y2))
print('\n\nThe encode Table values are:\n\n')
for i, item in enumerate(le.classes_):
    print(item, "-->", i)
print('\n\n')
start = time.time()

#Define Path
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'


#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)
le = LabelEncoder()
max_pad_len=174
warnings.warn = warn
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    print(mfcc.shape)
    return mfcc

def print_prediction(file_name):
    prediction_feature = extract_features(file_name)
    prediction_feature = prediction_feature.reshape(1, 40,174,1)

    predicted_vector = model.predict_classes(prediction_feature)
    print(predicted_vector)
    # predicted_class = le.inverse_transform(predicted_vector)
    print("\n\n\n\nThe predicted class is: ",predicted_vector , '\n\n\n\n')

filename = './Data/test/7383-3-1-0.wav'
print_prediction(filename)
# predicted_class[0]