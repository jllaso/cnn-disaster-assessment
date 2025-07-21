import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import InceptionResNetV2, ResNet50
from tensorflow.keras import applications, losses, optimizers, metrics
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from matplotlib.pyplot import imshow
import math
from numpy import expand_dims

# Descomprimir archivos
archive = zipfile.ZipFile('/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/Dataset/train.zip')
for file in archive.namelist():
    if file.startswith('cmasks/large/'):
        archive.extract(file, '/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train')
        print(file)

def maximum(a, b, c):
    return max([a, b, c])

# Rutas de imagen y segmentación
dir_seg_train = '/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train/cmasks/large'
dir_img_train = '/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train/cimages/large'
images_train = os.listdir(dir_img_train)
segmentations = os.listdir(dir_seg_train)

# Ajustar ruta de imágenes a JPG
dir_img_train = '/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train/cimages/jpg'
print(os.listdir(dir_img_train))

# Preparar datasets
img_pre_non_damaged = tf.data.Dataset.list_files('/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train/cimages/jpg/non_damaged/*predisaster.jpg', shuffle=False)
print(len(img_pre_non_damaged))
img_post_non_damaged = tf.data.Dataset.list_files('/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train/cimages/jpg/non_damaged/*postdisaster.jpg', shuffle=False)
img_pre_damaged = tf.data.Dataset.list_files('/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train/cimages/jpg/damaged/*predisaster.jpg', shuffle=False)
print(len(img_pre_damaged))
img_post_damaged = tf.data.Dataset.list_files('/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train/cimages/jpg/damaged/*postdisaster.jpg', shuffle=False)

img_pre = img_pre_damaged.concatenate(img_pre_non_damaged)
img_post = img_post_damaged.concatenate(img_post_non_damaged)

Y_damage = np.load("/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train/twoclasses/Y_damage.npy")
Y_no_damage = np.load("/content/drive/MyDrive/Chicago JLLF/Research Project/Proyecto/train/twoclasses/Y_no_damage.npy")
print(len(Y_no_damage))
print(len(Y_damage))

Y = np.concatenate((Y_damage, Y_no_damage), axis=0)
print(len(Y))

print(len(os.listdir(dir_img_train)))

def preprocess(filepath):
    print(filepath)
    byteimg = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(byteimg)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img

dataset = tf.data.Dataset.zip((img_pre, img_post, tf.data.Dataset.from_tensor_slices(Y)))

def preprocess_twin(pre_img, post_img, y):
    return (preprocess(pre_img), preprocess(post_img), y)

data = dataset.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

traindata = data.take(round(len(data) * 0.8)).batch(1602)
print(len(traindata))
testdata = data.skip(round(len(data) * 0.8)).batch(401)
print(len(testdata))

resnet_model = ResNet50(weights="imagenet")
inception_model = InceptionResNetV2(weights="imagenet")

def make_embedding():
    inp = Input(shape=(224, 224, 3), name='input_image')
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D((2, 2), padding='same')(c1)
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D((2, 2), padding='same')(c2)
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D((2, 2), padding='same')(c3)
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    o1 = Dropout(0.4)(f1)
    d1 = Dense(4096, activation='sigmoid')(o1)
    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding_model = make_embedding()
embedding_model.summary()

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, pre_embedding, post_embedding):
        return tf.math.abs(pre_embedding - post_embedding)

def make_siamese_model():
    input_pre_img = Input(name='inpre', shape=(224, 224, 3))
    input_post_img = Input(name='inpost', shape=(224, 224, 3))
    siamese_layer = L1Dist(name='distance')
    distances = siamese_layer(embedding_model(input_pre_img), embedding_model(input_post_img))
    damage_score = Dense(1, activation='sigmoid')(distances)
    return Model(inputs=[input_pre_img, input_post_img], outputs=damage_score, name='SiameseNetwork')

siamese_model = make_siamese_model()
siamese_model.summary()

binary_crossentropy_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-5)

train_pre, train_post, y = traindata.as_numpy_iterator().next()
y_train = y
x_train = [train_pre, train_post]

test_pre, test_post, y_true = testdata.as_numpy_iterator().next()
y_test = y_true
x_test = [test_pre, test_post]

siamese_model.compile(opt, binary_crossentropy_loss, metrics=['accuracy'])
hist = siamese_model.fit(x_train, y_train, batch_size=16, validation_data=(x_test, y_test), epochs=50, shuffle=True, verbose=1)
