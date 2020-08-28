'''
@author:Amal Mathew

'''
import pandas as pd
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import tensorflow as tf

max_pad_len=174
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    print(mfcc.shape)
    return mfcc


fulldatasetpath = './Data/audio/'
metadata = pd.read_csv('./Data/metadata/UrbanSound8K.csv')
features = []
features2 = []

for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath), 'fold' + str(row["fold"]) + '/',
                             str(row["slice_file_name"]))
    class_label = row["class"]
    classid = row["classID"]
    data = extract_features(file_name)
    features.append([data, class_label])
    features2.append([data, class_label, classid])

featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
featuresdf2 = pd.DataFrame(features2, columns=['feature', 'class_label', 'classid'])
# print('Finished feature extraction from ', len(featuresdf), ' files')
# print('Finished feature extraction from ', len(featuresdf), ' files')

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
# # X=X.reshape(X.shape[0], 40, 174, 1)
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

# x_train=x_train.reshape(x_train.shape[0],40,174,1)
x_train=x_train.reshape(x_train.shape[0], 40, 174, 1)
# x_new_test=np.array(features_new_test)
x_test=x_test.reshape(x_test.shape[0], 40, 174, 1)
print('entering model')

model=tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(40,174,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    # tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

from datetime import datetime

#num_epochs = 12
#num_batch_size = 128

num_epochs = 120
num_batch_size = 256

checkpointer=tf.keras.callbacks.ModelCheckpoint(filepath='weights.best.basic_cnn.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

import os
target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')

