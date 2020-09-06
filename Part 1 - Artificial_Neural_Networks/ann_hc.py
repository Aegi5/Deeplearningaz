#artificial neural network

#Partie 1 : préparation des données 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# epoch : An epoch is one complete presentation of the data set to be learned to a learning machine. 

#encodage des variables catégories
#from sklearn.compose import ColumnTransformer
# Encode categorical data and scale continuous data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
preprocess = make_column_transformer(
        (OneHotEncoder(), ['Geography', 'Gender']),
        (StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance',
                            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                            'EstimatedSalary']))
X = preprocess.fit_transform(X)
X = np.delete(X, [0,3], 1)
#on encode le pays suivant des colones avec des variables à 1 ou 0 selon l'appartenant des individus aux pays
#pas besoin pour le genre qui sera 0 ou 1


#problème avec la new version categorical_features
#plus besoin a priori de LabelEncoder 
#ct = ColumnTransformer([("Country", OneHotEncoder(), [1])],    remainder = 'passthrough')
#X = ct.fit_transform(X)
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""  #deja fait plus au dans le preprocess





#Partie 2 : Construction du réseau de neurones

#importation des modules de keras
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense


#Initialisation
#Attention ! les instances d'objets et fonctions de tensorflow.keras et keras ne sont pas compatibles entre elles 
#plus maintenenant car tf est devenu le backend par defaut de keras ?
from keras.layers import Dropout
classifier = Sequential()

#Ajout de la couche d'entrée et d'une couche cachée
classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer='uniform',
                     input_dim=11)) # creation de la couche cachée ici on prend la moyenne entre le nombre de neurones en entrée (qui correspond au nombre de variables d'entrée) ici 11
#et le nombre de variables de sorti ici 1, celan n'est pas une règle ! on prend ce critère arbitrairement
# idéalement il faudrait essayer different nombre de neurore et comparer les résultats entre eux
#il faut spécificer la dim de l'input layer


 classifier.add(Dropout(rate=0.1)) 
 #rate : chaque neurone a rate chance de se desactiver
 #augmenter le rate si on a un problème de surapprentissage



#Ajout d'une deuxième couche cachée
classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer='uniform'))

#Ajout de la couche de sortie
classifier.add(Dense(units=1, activation="sigmoid", 
                     kernel_initializer='uniform')) #1 seul neurone car on a prédire qu'une seule valeur qui déterminera si le client va rester ou quitter la banque
#on utilise une sigmoid pour obtenir une proba pour classifier la sortie en 2 catégories : reste ou part de la banque
#si 3 catégories -> utilisation du soft_max qui correspond à une sigmoid appliquée à plus de 2 catégories

#compiler le reseau de neurones
classifier.compile(optimizer="adam",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
#binary_cross_entropy pour 2 catérgories en sortie
#categorical pour plus de 2
#metrics : déf la grandeur mesurant les performances

#entrainer le reseau de neurones
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


#courbes accuracy / loss
"""
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""






#PARTIE 3
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Besoin d'un seuil pour décider à partir d'une proba si l'individu reste ou pars

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#####faire une prediction pour un individu
test = {"RowNumber" : None, "CustomerId" : None,"Surname": None,"CreditScore" : 600, "Geography":"France", "Gender":"Male","Age":40,"Tenure":3, "Balance":60000, "NumOfProducts":2, "HasCrCard":1,  "IsActiveMember":1, "EstimatedSalary" :50000, "Exited":None}

newprediction = classifier.predict(sc.transform(np.array([[0, 0.0, 600, 0, 40.0, 3.0, 60000.0, 2.0, 1.0, 1.0, 50000.0]])))
newprediction = (newprediction > 0.5)


#On suppose le réseau de neurones déjà construit et entraîné
# Predicting à la dernière ligne
y_pred = classifier.predict(X)
y_pred = (y_pred > 0.5)

#pour un individu
"""
#pour un indivu
Xnew = pd.DataFrame(data={
        'CreditScore': [600], 
        'Geography': ['France'], 
        'Gender': ['Male'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [50000]})
Xnew = preprocess.transform(Xnew)
Xnew = np.delete(Xnew, [0,3], 1)
new_prediction = classifier.predict(Xnew)
new_prediction = (new_prediction > 0.5)
"""



"""
k-fold cross validation :
résoue le prb de variance
au lieu de scinder le dataset en deux parties training set et test set 
on sépare le dataset en 10 sous lots chacun sous-divisé en 10 tas
chaque lot correspond à une itération
pour un lot donnée on chosit au hasard un tas parmi les 10 qui sera notre set de set et les autres les set d'entrainement

biais élevé -> prédictions sont loin des véritables valeurs
variance faible ->  imprécision consistante à traves les différents entrainements

biais faible ->  prédictions est assez précise
variance élevée -> précision des prédictions varie bcp au fil des entrainements

biais élevé
variance élevée -> prédictions pas bonnes, précision bonnes et mauvaises au fil des entrainements


objectif :
biais fable : précision ne sont pas éloignées de la "moyenne" qui est la valeur visée -> précise
variance faible : indique que même si l'on réentraîne le modèle plusieurs fois, peu de variation de la précision, erreur consistante


"""


from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score


def build_classifier():
    
    classifier = Sequential()
#ajout d'une couche d'entrée et d'une couche cachée
    classifier.add(Dense(units=6, activation="relu", 
                         kernel_initializer='uniform',
                         input_dim=11))

    classifier.add(Dropout(rate=0.1)) 
    
#Ajout d'une deuxième couche cachée
    classifier.add(Dense(units=6, activation="relu", 
                         kernel_initializer='uniform'))

    #Ajout de la couche de sortie
    classifier.add(Dense(units=1, activation="sigmoid", 
                         kernel_initializer='uniform')) 
    #compiler le reseau de neurones
    classifier.compile(optimizer="adam",
                       loss="binary_crossentropy",
                       metrics=["accuracy"])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)  #transforme notre classifier pour qu'il soit adaptable avec cross_val_score, le classifier etait une instance d'une classe de keras et en instanciant un object de type kerasclassifier prenant le dernier en arg on le rend
# compatible avec sklearn
#build_fn pour build fonction et les autres arg sont ceux utilisés pour parametrer l'entrainement

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, 
                             cv = 10, n_jobs = 1)



#mesure de précision
#sortie de cross_val_score = 10 valeurs de précision
#ici cv=10 pour créer 10 grp qui sont suffisant pour avoir une idée de la précision et de la variance du modèle ici
#n_jobs pour spécifier le nombre de coeurs à utiliser pour paralléliser les calculs, -1 pour utiliser tous les coeurs, ne marche pas si cv != 1 et je ne sais pas pourquoi

moyenne = accuracies.mean()
variance = accuracies.std()


"""regularisation dropout pour résoudre le problème de surapprentissage
visible quand le modèle n'arrive pas à généraliser ses acquis lors des test
indicateurs : precision très elevées sur le jeu d'entrainement par rapport à celle sur le jeu de test
Le décrochage, ou abandon, est une technique de régularisation pour réduire le surajustement dans les réseaux de neurones. La technique évite des co-adaptations complexes sur les données de l'échantillon d'entraînement. C'est un moyen très efficace d'exécuter
un moyennage du modèle de calcul avec des réseaux de neurones1. Le terme "décrochage" se réfère à une suppression temporaire de neurones (à la fois les neurones cachés et les neurones visibles) dans un réseau de neurones.
Le réseau neuronal se voit amputé d'une partie de ses neurones pendant la phase d'entrainement (leur valeur est estimée à 0) et ils sont par contre réactivés pour tester le nouveau modèle. 

"""




"""
hyperparamètre :
In machine learning, a hyperparameter is a parameter whose value is set before the learning process begins. By contrast, the values of other parameters are derived via training.

Hyperparameters can be classified as model hyperparameters, that cannot be inferred while fitting the machine to the training set because they refer to the model selection task, or algorithm hyperparameters,
that in principle have no influence on the performance of the model but affect the speed and quality of the learning process. An example of the first type is the topology and size of a neural network.
An example of the second type is learning rate or mini-batch size.

Different model training algorithms require different hyperparameters, some simple algorithms (such as ordinary least squares regression) require none. Given these hyperparameters, the training algorithm learns the parameters from the data. For instance,
LASSO is an algorithm that adds a regularization hyperparameter to ordinary least squares regression, which has to be set before estimating the parameters through the training algorithm. 

"""

#Partie 4

#Optimiser les hyperparametres algo : gridsearch recherche en grille
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    
    classifier = Sequential()
#ajout d'une couche d'entrée et d'une couche cachée
    classifier.add(Dense(units=6, activation="relu", 
                         kernel_initializer='uniform',
                         input_dim=11))
    classifier.add(Dropout(rate=0.1)) 
    
#Ajout d'une deuxième couche cachée
    classifier.add(Dense(units=6, activation="relu", 
                         kernel_initializer='uniform'))
    classifier.add(Dropout(rate=0.1)) 

    #Ajout de la couche de sortie
    classifier.add(Dense(units=1, activation="sigmoid", 
                         kernel_initializer='uniform')) 
    #compiler le reseau de neurones
    classifier.compile(optimizer=optimizer,
                       loss="binary_crossentropy",
                       metrics=["accuracy"])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)  #on supprime batch_size et epoch pour les faire varier
#gridsearch va jouer sur les paramètres à optimiser
#il nécessite un dictionnaire dont les clés seront les paramètres à faire varier et les elements les valuers à tester
parameters = {"batch_size" : [25,32,64], #en general les puissances de 2 fonctionnent bien
              "epochs":[100,300], 
              "optimizer":["adam", "rmsprop"]}
    
grid_search = GridSearchCV(estimator=classifier, 
                          param_grid=parameters, 
                          scoring="accuracy",
                          cv=10) #pour chaque combi d'hyperparameters on effectue une k-fold x validation pour estimer la précision

grid_search = grid_search.fit(X_train, y_train)


best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_

# best_param =  {'batch_size': 32, 'epochs': 300, 'optimizer': 'rmsprop'}
# best_prec = 0.841


    