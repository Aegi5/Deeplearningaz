# -*- coding: utf-8 -*-

#Model from https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')


# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)

#evaluation
model.evaluate(x_test, y_test)

#prediction
image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())



#Mine :


#Etape 1

#comment structurer les données ?
#jeux d'entrainement dans 2 dossiers differents dont le nom correpond à la catégorie d'image qu'ils contiennent
#les fichiers images sont nommés avec leur catégorie suivie d'une numérotation
# Feature Scaling

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()


# Reshaping the array to 4-dims so that it can work with the Keras API
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_test /= 255

#Etape 2 : construction du CNN

#Modules
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Sequential : module servant à initializer un NN
#Rappel : il y a deux manière d'initializer un NN, soit de manière sequentiel soit avec un graphe.
#Ici on utilise une sequence de couche donc usage de Sequential

#Convolution2D : sert pour effectuer des convolutions, ici les entrées sont des images donc en 2D, les videos sont en 3D (2D + temps)

#MaxPooling2D: sert à appliquer le MaxPooling et 2D car analyse d'image

#Flatten : pour applatir les features maps en un vecteur

#Dense : pour ajouter des couches totalement co

#Etape 3 - Initialiser un CNN
classifier = Sequential()



#Etape 4 - Ajout de la couche de convolution
#choix du nombre de feature dectectors (et du coup du nb de feature map créée)
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
                                          input_shape=input_shape,
                                          activation="relu"))

#Etape 4 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
"""
pool_size : taille du kernel, 2x2 est un bon compromis pour réduire la complexité du modèle en /2 la taille de la feature map et de conserver assez d'info pertinente
Strides values. If None, it will default to pool_size.
"""

#ajout d'une couche de convo + pooling pour ameliorer la precision
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
                                          activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

"""
input_shape à changer : puisque les images en entrée de la couche sortent de la couche de convo et de pooling et que cette dernière /2 la taille (car kernel de 2x2)
la taille des images s'approche de 32x32 mais pas exactement, ici comme la couche est imbriquée dans le reseau et il s'adopte automatique à l'input shape ainsi on peut supprimer le paramètre relative à la taille
"""

#Etape 5 - Flattening
"""
pourquoi ne pas avoir fait depuis le flattening ? -> car perte de la structure de l'image càd des caractéristiques importantes que l'on ne peut detecter qu'avec son voisinnage
"""
classifier.add(Flatten())


#Etape 4 - Ajout du NN fully connected pour extraire des associations, des patterns et faire des predictions non linear
#couche cachée 
classifier.add(Dense(units=128, activation="relu")) 
"""
Idem le nombre de input est automatiquement adapté
"""

#couche de sortie
classifier.add(Dense(units=10, activation="sigmoid"))
"""
Combien de neurones prendre ? Moyenne nb de neurone couche d'entree + couche de sortie ?
ici la couche de sortie contient 10 neurones (qui donne une proba) car on cherche à classifier des images selon les chiffres
Ici, question épineuse car on effectue du pooling et du flattening
On prend un nombre ni trop petit ni trop grand(q de puissance) et puissance de 2 
"""


#Etape 5 : Compilation
#choix de l'algo du gradient, de la fonction de cout, de la metrics pour mesurer la performance
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])




#Etape 6 : Fitting images on NN  + entrainement de CNN

classifier.fit(x=X_train,y=Y_train, epochs=10)


#res : loss: 0.0075 - acc: 0.9973
#exemple 
import matplotlib.pyplot as plt
image_index = 7777 # You may select anything up to 60,000
print(Y_train[image_index]) # The label is 8
plt.imshow(X_train[image_index], cmap='Greys')

#prediction
image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
