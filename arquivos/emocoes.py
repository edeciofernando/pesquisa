import cv2
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# abertura dos pesos salvos
with open('C:\\00 Projeto Pesquisa\\reconhecimento_emocoes\\network_emotions.json', 'r') as json_file:
    json_saved_model = json_file.read()
json_saved_model
# acessa o arquivo com a rede neural
network_loaded = tf.keras.models.model_from_json(json_saved_model)
# acessa o arquivo com os pesos
network_loaded.load_weights('C:\\00 Projeto Pesquisa\\reconhecimento_emocoes\\weights_emotions.hdf5')
# compila a rede neural
network_loaded.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# para confirmar se a rede neural foi carregada corretamente:
# network_loaded.summary()

# classificar imagens pela webcam:
captura = cv2.VideoCapture(0)
conectado, video = captura.read()  # faz a leitura do primeiro frame do video
detector_face = cv2.CascadeClassifier('C:\\00 Projeto Pesquisa\\reconhecimento_emocoes\\haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# print(conectado, video.shape)

# processar cada um dos frames do vídeo:
while (cv2.waitKey(1) < 0):
    conectado, frame = captura.read()
    if not conectado:
        break
    deteccoes = detector_face.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
    if len(deteccoes) > 0:
        for (x, y, w, h) in deteccoes:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # desenha o bounding box
            # fazer a previsão:
            roi = frame[y:y + h, x:x + w]  # extrai o ROI (region of interest)
            roi = cv2.resize(roi, (48, 48))  # redimensionamento
            roi = roi / 255  # normalização dos pixels
            roi = np.expand_dims(roi, axis=0)  # expande as dimensões
            previsao = network_loaded.predict(roi)  # busca a previsão

            if previsao is not None:
                resultado = np.argmax(previsao)
                # coloca o texto no frame que está sendo processado:
                cv2.putText(frame, emotions[resultado], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)

    cv2.imshow('Video', frame)

# When everything is done, release the capture
captura.release()
cv2.destroyAllWindows()
