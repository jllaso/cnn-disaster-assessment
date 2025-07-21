# ── siamese_pipeline.py ──

import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import ResNet50, InceptionResNetV2
from tensorflow.keras import applications, losses, optimizers, metrics
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from PIL import Image

# Definir rutas
ARCHIVE_PATH = '/content/drive/MyDrive/Chicago JLLF/ResearchProject/Proyecto/Dataset/train.zip'
EXTRACT_TO = '/content/drive/MyDrive/Chicago JLLF/ResearchProject/Proyecto/train'
SEG_SUBDIR = 'cmasks/large/'
DIR_SEG_TRAIN = os.path.join(EXTRACT_TO, SEG_SUBDIR)
DIR_IMG_TRAIN = os.path.join(EXTRACT_TO, 'cimages/large')
# Ajustar si necesitas cambiar la carpeta a jpg
DIR_IMG_TRAIN = os.path.join(EXTRACT_TO, 'cimages/jpg')

# ── BLOQUE 2: Extracción del zip y funciones auxiliares ──

# Extraer solo las máscaras grandes desde el archivo zip
with zipfile.ZipFile(ARCHIVE_PATH) as archive:
    for file in archive.namelist():
        if file.startswith('cmasks/large/'):
            archive.extract(file, EXTRACT_TO)
            print(f"Extracted: {file}")

# Función para obtener el máximo de tres valores
def maximum(a, b, c):
    return max([a, b, c])

# Listar imágenes y máscaras
images_train = os.listdir(DIR_IMG_TRAIN)
segmentations = os.listdir(DIR_SEG_TRAIN)

print(f"Número de imágenes: {len(images_train)}")
print(f"Número de máscaras: {len(segmentations)}")

# ── BLOQUE 3: Cargar datasets tf.data.Dataset y arrays Y ──

# Rutas de imágenes clasificadas
IMG_DIR = '/content/drive/MyDrive/Chicago JLLF/ResearchProject/Proyecto/train/cimages/jpg'
img_pre_non_damaged = tf.data.Dataset.list_files(f'{IMG_DIR}/non_damaged/*predisaster.jpg', shuffle=False)
img_post_non_damaged = tf.data.Dataset.list_files(f'{IMG_DIR}/non_damaged/*postdisaster.jpg', shuffle=False)
img_pre_damaged = tf.data.Dataset.list_files(f'{IMG_DIR}/damaged/*predisaster.jpg', shuffle=False)
img_post_damaged = tf.data.Dataset.list_files(f'{IMG_DIR}/damaged/*postdisaster.jpg', shuffle=False)

# Concatenar pre y post
img_pre = img_pre_damaged.concatenate(img_pre_non_damaged)
img_post = img_post_damaged.concatenate(img_post_non_damaged)

# Cargar etiquetas
Y_damage = np.load('/content/drive/MyDrive/Chicago JLLF/ResearchProject/Proyecto/train/twoclasses/Ydamage.npy')
Y_no_damage = np.load('/content/drive/MyDrive/Chicago JLLF/ResearchProject/Proyecto/train/twoclasses/Ynodamage.npy')

Y = np.concatenate((Y_damage, Y_no_damage), axis=0)

print(f"Total etiquetas: {len(Y)}")

# ── BLOQUE 4: Preprocesamiento y ensamblado del dataset ──

def preprocess(file_path):
    print(file_path)
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img

# Dataset de entrada: (img_pre, img_post, etiqueta)
dataset = tf.data.Dataset.zip((img_pre, img_post, tf.data.Dataset.from_tensor_slices(Y)))

# Aplicar preprocesamiento a cada pareja de imágenes
def preprocess_twin(pre_img_path, post_img_path, label):
    return (preprocess(pre_img_path), preprocess(post_img_path), label)

data = dataset.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Dividir en train y test
train_size = int(len(Y) * 0.8)
train_data = data.take(train_size).batch(1602)
test_data = data.skip(train_size).batch(401)

print(f"Tamaño train: {len(train_data)}")
print(f"Tamaño test: {len(test_data)}")

# ── BLOQUE 5: Modelos de embedding ──

# Custom CNN embedding
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
    
    return Model(inputs=inp, outputs=d1, name='embedding')

# Embedding con ResNet50
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
def make_embedding_resnet():
    return resnet_model

# Embedding con InceptionResNetV2
inception_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
def make_embedding_inception():
    return inception_model

# Instanciar modelos
embedding_model = make_embedding()
embedding_model_resnet = make_embedding_resnet()
embedding_model_inception = make_embedding_inception()

embedding_model.summary()
print(f"Número de capas ResNet: {len(embedding_model_resnet.layers)}")
print(f"Número de capas Inception: {len(embedding_model_inception.layers)}")

# ── BLOQUE 6: Definición de modelos siameses ──

# Distancia L1
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, pre_embedding, post_embedding):
        return tf.math.abs(pre_embedding - post_embedding)

# Modelo siamés custom
def make_siamese_model():
    input_pre = Input(name='in_pre', shape=(224, 224, 3))
    input_post = Input(name='in_post', shape=(224, 224, 3))

    distances = L1Dist()(embedding_model(input_pre), embedding_model(input_post))
    damage_score = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_pre, input_post], outputs=damage_score, name='SiameseNetwork')

# Modelo siamés con ResNet
def make_siamese_model_resnet():
    input_pre = Input(name='in_pre', shape=(224, 224, 3))
    input_post = Input(name='in_post', shape=(224, 224, 3))

    distances = L1Dist()(embedding_model_resnet(input_pre), embedding_model_resnet(input_post))
    damage_score = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_pre, input_post], outputs=damage_score, name='SiameseNetworkResNet')

# Modelo siamés con Inception
def make_siamese_model_inception():
    input_pre = Input(name='in_pre', shape=(299, 299, 3))
    input_post = Input(name='in_post', shape=(299, 299, 3))

    distances = L1Dist()(embedding_model_inception(input_pre), embedding_model_inception(input_post))
    damage_score = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_pre, input_post], outputs=damage_score, name='SiameseNetworkInception')

# Instanciar modelos siameses
siamese_model = make_siamese_model()
siamese_model_resnet = make_siamese_model_resnet()
siamese_model_inception = make_siamese_model_inception()

siamese_model.summary()

# ── BLOQUE 7: Compilación y entrenamiento ──

# Pérdida y optimizador
binary_crossentropy_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-5)

# Extraer un batch para pruebas iniciales
train_pre, train_post, y_train = train_data.as_numpy_iterator().next()
test_pre, test_post, y_test = test_data.as_numpy_iterator().next()
x_train = [train_pre, train_post]
x_test = [test_pre, test_post]

# Entrenar modelo siamés custom
siamese_model.compile(optimizer=opt, loss=binary_crossentropy_loss, metrics=['accuracy'])
hist = siamese_model.fit(
    x_train, y_train,
    batch_size=16,
    validation_data=(x_test, y_test),
    epochs=50,
    shuffle=True,
    verbose=1
)

# Entrenar modelo siamés ResNet
siamese_model_resnet.compile(optimizer=opt, loss=binary_crossentropy_loss, metrics=['accuracy'])
hist_resnet = siamese_model_resnet.fit(
    x_train, y_train,
    batch_size=16,
    validation_data=(x_test, y_test),
    epochs=50,
    shuffle=True,
    verbose=1
)

# Entrenar modelo siamés Inception
siamese_model_inception.compile(optimizer=opt, loss=binary_crossentropy_loss, metrics=['accuracy'])
hist_inception = siamese_model_inception.fit(
    x_train, y_train,
    batch_size=16,
    validation_data=(x_test, y_test),
    epochs=50,
    shuffle=True,
    verbose=1
)

