from imutils import paths
import cv2
from helpers import resize_to_fit
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from matplotlib import pyplot as plt

letter_images = "extracted_letters"
model_pics = "simple_captcha_model.hdf5"
model_labels = "model_labels.dat"

data = []
labels = []

for image_file in paths.list_images(letter_images):

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = resize_to_fit(image, 20, 20)

    image = np.expand_dims(image, axis=2)

    label = image_file.split(os.path.sep)[-2]

    data.append(image)
    labels.append(label)


data = np.array(data, dtype=float) / 255.0
labels = np.array(labels)

(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.2, random_state=0)

lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

with open(model_labels, "wb") as f:
    pickle.dump(lb, f)

model = Sequential()

model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500, activation="relu"))

model.add(Dense(32, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=12, verbose=1)

model.save(model_pics)

print(history)

a = plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
a.show()

b = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
b.show()
