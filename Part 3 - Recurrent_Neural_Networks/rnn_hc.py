# -*- coding: utf-8 -*-


#Recurrent Neural Network

# Partie 1 : préparation des données

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#jeu d'entrainement
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train[["Open"]].values #pour aller des colonnes dans le dataframe [["colonne1", "colonne2"]]  , .values (dataset to array)


#feature scaling 
"""
standardisation : x - means / standard dev x
normalisation : x - min x / (max x - min x) -> values between [0;1] -> préférable à utiliser lors de fonction d'activation sigmoide
"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
training_set_scaled = sc.fit_transform(training_set)


#creation de la structure avec 60 timesteps et 1 sortie
"""
prédiction d'une valeur à l'aide des valeurs des 60 derniers jours
why 60 ? 1 timestep (1 jour) -> surapprentissage
"""
x_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[(i-60):i, 0])
    y_train.append(training_set_scaled[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)


#reshaping
"""
ajout d'une dimension pour empiler des matrices, ici on pourrait prendre une matrice avec la valeur des actions de samsung dont les profits sont corrélés à ceux de google
les fonctions qui suivent nécessitent des matrices 3D (la couche de RNN) mais la doc ne précise pas les dims, la vidéo est pas à jour
"""
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Partie 2 : construction du RNN

#libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialisation
#prediction de valeur continue = régréssion
regressor = Sequential()


#Couche LSTM + Dropout
"""
ici on cherche à prédire le cours d'une action,une tendance, trouver un nb de neurones suffisamment conséquent mais pas trop de tps de calcul
(augmenter le nombre de LSTM en fonction de la complexité du prb ?)
accumulation de couches LSTM : true
"""
regressor.add(LSTM(units=50, return_sequences= True, 
                   input_shape=(x_train.shape[1], 1)))
"""input_shape = taille des données en entrée, dépend de x_train
or le LSTM ne s'interesse pas ici aux nombres d'observations (autant que de jours) mais plutôt aux timesteps et le nombre de prédicteurs ou le nombre de variable (la valeur d'une action à prédire)
"""
#Dropout layer diminishes the overfiting
regressor.add(Dropout(rate=0.2))


# couche 2 :LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True)) #plus besoin de spécifier d'input shape car déjà spécifié plus haut
regressor.add(Dropout(0.2))

# couche 3 :LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True)) 
regressor.add(Dropout(0.2))

# couche 4:LSTM + Dropout
regressor.add(LSTM(units=50)) #dernière couche LSTM, return_sequence = False (valeur par def)
regressor.add(Dropout(0.2))

#couche de sortie
regressor.add(Dense(units=1))

#compilation
#optimizer (type d'algo du grad) ET cost function
regressor.compile(optimizer="adam", loss="means_squared_error")

#ici on prend l'optimizer adam (car après test il convient le mieux, un autre choix pour les RNNs aurait été RMSProp)
#de manière générale, si l'on ne sait pas quel optimizer prendre, chosir Adam par défaut
#ici, prbp de régression, on utilse moindre carrés


#training
regressor.fit(x_train, y_train, epochs=100, batch_size=32)

# Partie 3 : Prédictions et visualisation


