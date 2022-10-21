import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

gerador_treinamento = ImageDataGenerator(rescale=1./255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         zoom_range=0.2)
dataset_treinamento = gerador_treinamento.flow_from_directory('C:\\00 Pesquisa\\fer2013\\train',
                                                              target_size = (48, 48),
                                                              batch_size = 16,
                                                              class_mode = 'categorical',
                                                              shuffle = True)

np.unique(dataset_treinamento.classes, return_counts=True)

gerador_teste = ImageDataGenerator(rescale=1./255)
dataset_teste = gerador_teste.flow_from_directory('C:\\00 Pesquisa\\fer2013\\validation',
                                                              target_size = (48, 48),
                                                              batch_size = 1,
                                                              class_mode = 'categorical',
                                                              shuffle = False)

numero_detectores = 32 # numero de filtros da rede neural convolucional
numero_classes = 7
largura, altura = 48, 48
epocas = 70

# construção da estrutura de uma rede neural do zero
network = Sequential()

network.add(Conv2D(filters=numero_detectores, kernel_size=(3,3), activation='relu', padding='same',
                   input_shape=(largura, altura, 3)))
network.add(BatchNormalization())
network.add(Conv2D(filters=numero_detectores, kernel_size=(3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

# DEFININDO MAIS UM BLOCO DE REDE NEURAL
network.add(Conv2D(filters=2*numero_detectores, kernel_size=(3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(filters=2*numero_detectores, kernel_size=(3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Conv2D(filters=2*2*numero_detectores, kernel_size=(3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(filters=2*2*numero_detectores, kernel_size=(3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

# DEFININDO MAIS UM BLOCO DE REDE NEURAL
network.add(Conv2D(filters=2*2*2*numero_detectores, kernel_size=(3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(filters=2*2*2*numero_detectores, kernel_size=(3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Flatten()) # CAMADA DE FLATTEN

network.add(Dense(units=2 * numero_detectores, activation='relu')) # CAMADA DENSA COM 64 NEURONIOS
network.add(BatchNormalization()) # CAMADA DE BATCH NORMALIZATION SOBRE A CAMADA DENSA
network.add(Dropout(0.2)) # CAMADA DE DROP OUT (zerando 20% dos neuronios da camada escondida - recomendado)

network.add(Dense(units=2 * numero_detectores, activation='relu')) # CAMADA DENSA COM 64 NEURONIOS
network.add(BatchNormalization()) # CAMADA DE BATCH NORMALIZATION SOBRE A CAMADA DENSA
network.add(Dropout(0.2)) # CAMADA DE DROP OUT (zerando 20% dos neuronios da camada escondida - recomendado)

# CAMADA DE SAIDA
network.add(Dense(units=numero_classes, activation='softmax'))

#print(network.summary())

network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TREINAMENTO
network.fit(dataset_treinamento, epochs=epocas)

# salva a rede neural
model_json = network.to_json()
with open('network_emotions.json', 'w') as json_file:
  json_file.write(model_json)

# salva os pesos da rede neural
from keras.models import save_model
network_saved = save_model(network, 'C:\\00 Pesquisa\\weights_emotions.hdf5')