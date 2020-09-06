#Data preprocessing

#loading packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #extension de numpy,utile pour manipuler des jeux de données

#loading data
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:,:-1].values #dataset.iloc[lines,columns]
y = dataset.iloc[:,-1].values


#loading missing data
from sklearn.preprocessing import Imputer #is deprecated gonna be remove use the next line instead

#from sklearn.impute import SimpleImputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0) #nan = not a number, typologie pour les données manquantes
#avec Imputer le paramètre axis=0 indiquait le remplissement par colonne
imputer = imputer.fit(X[:,1:3])#méthode fit donne le jeu de données pour que l'imputer fasse les calculs, on lui donne ici que les colonnes ou les données sont manquantes
X[:,1:3] = imputer.transform(X[:, 1:3])


#encodage des variables catégories
#encode les variables indépendantes (predictors)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #transforme les variables catégoriques en nombre
labelencoder_X =  LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X= onehotencoder.fit_transform(X).toarray()

labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)


#separating training dataset and testing dataset
from sklearn.model_selection import train_test_split #not in from sklearn.cross_validation import train_test_split anymore
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)


#changing scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)#calcul la moyenne et l'ecart-type puis transforme càd applique (donnée-avg /ecart-type) aux données
X_test = sc_X.transform(X_test)#standarize the values using the same -avg /ecart-type used before ! on utilise transform au lieu de fit_transform pour ne pas avoir a recalculer les moyennes et ecart-types précedemment calculés


#ici cas de classification : le client va-t-il acheter ou non le produit ? Que les valeurs aillent de 0 à 1 n'importe pas du moment que l'on peut
#faire la différence entre les deux
#donc pas besoin de standardiser y










