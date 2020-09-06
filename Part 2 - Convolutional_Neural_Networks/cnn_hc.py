# -*- coding: utf-8 -*-

"""
Article : Understanding Convolutional Neural Networks with AMathematical Mode
https://arxiv.org/pdf/1609.04112.pdf

Etape 1
Convolution ??
En mathématiques, le produit de convolution est un opérateur bilinéaire et un produit commutatif, 
généralement noté « ∗ », qui, à deux fonctions f et g sur un même domaine infini, 
fait correspondre une autre fonction « f ∗ g » sur ce domaine, 
qui en tout point de celui-ci est égale à l'intégrale sur l'entièreté du domaine (ou la somme si celui-ci est discret) d'une des deux fonctions autour de ce point, pondérée par l'autre fonction autour de l'origine — les deux fonctions étant parcourues 
en sens contraire l'une de l'autre (nécessaire pour garantir la commutativité). 

Le produit de convolution généralise l'idée de moyenne glissante et est la représentation mathématique de la notion de filtre linéaire. 
Il s'applique aussi bien à des données temporelles (en traitement du signal par exemple) qu'à des données spatiales (en traitement d'image). 
En statistique, on utilise une formule très voisine pour définir la corrélation croisée. 
cf wikipedia pour la definition formelle et les applications

De même :
La convolution est le processus consistant à ajouter chaque l'élément de l'image à ses voisins 
immédiats, pondéré par les éléments du noyau. C'est une forme de produit de convolution.
Il devra être noté que les opérations matricielles effectuées (les convolutions) ne sont 
pas des multiplications traditionnelles de matrices malgré le fait que ce soit noté par un « * ». 
https://fr.wikipedia.org/wiki/Noyau_(traitement_d%27image)
https://www.f-legrand.fr/scidoc/docimg/image/filtrage/convolution/convolution.html
Opérateur de convolution : x entouré d'un rond

Input : image
Feature detector/Kernel(ou noyau)/Filtre : grille/matrice carrée usuellement de 3x3 (d'autres tailles sont possibles)
le kernel lors de son deplacement peut ne pas ê totalement rempli càd comprend une zone en-dehors de l'image
On fait glisser le kernel sur toute l'image.
Stride ou foulée : mesure de déplacement du kernel sur une image, peut se faire de gauche à droite et de haut en bas d'un pixel par exemple
Feature map : resultat de la convolution de l'image et du feature detector pour une stride donnée
    -> le resultat contient moins de données et donc réduit la taille de l'image/nb de données
    -> le Feature dectector permet de detecter les features qui correspondent aux valeurs les plus elevées dans la feature map
    -> l'idée est de retirer de l'image d'origine les parties ne contenant pas de feature et reperer les endroits où la feature apparait et de conserver l'info
    -> Feature dectector est un filtre : ne garde que la partie d'interêt et se débaresse du reste
    
    -> le reseau cherche lors de l'entrainement à creer des features detector et à l'aide des features maps générés et par comparaison aux images annotées permettre l'identification des features
    -> exple : lors de l'entrainement pour la reconnaissance d'un nez ou d'un visage, pour chacune des features que le reseau a repéré il va creer une feature map et un feature dector capable de nous dire où se trouve les features qu'il aura detecté
Convolutional layer : correspond à l'ensemble des resultats de l'application des convolutions entre les inputs et les features detectors








Etape 1B
Couche ReLU:
Précédemment : creation de plein de feature maps à partir de l'image d'origine 
on applique sur celles-ci la fonction redresseur
objectif : rendre le modèle davantage non linéaire (créer )
les images sont déjà non linéaires (contient des pixels RGB differents côte à côte, des contours). L'application seule de la fonction convolution, on prend le risque de créer une sortie linéaire



Etape 2
Pooling ou Downsampling
Max pooling :
Le reseaux de neurones doit avoir la propriete d'invariance spatiale càd que quelque soit la position, la taille, l'orientation de la feature, le reseau doit savoir la retrouver

Feature map ---> operation Max pooling ---->  Pooled feature map

Processus similaire à la convolution, on se donne un kernel qui glisse sur la feature map et on extrait pour une position donnée du kernel le max des valeurs englobées dans le kernel et on l'inscrit dans la pooled feature map
en général, stride = 2
-> on "garde" une idée de la position et des valeurs des features càd des info importantes
-> on a retiré de l'information/des pixels mais seulement celle qui ne contient pas la feature et conservé celle qui la contient
-> on remarque que même si on deplace un peu la feature sur la feature map, on aurait une meme sortie
-> la suppression des informations peu interessantes car moins de données à traiter et permet d'eviter un surentrainement sur celles-ci et celles d'intérêt servent à généraliser le modèle
Pooling layer: couche correspond au resultat de l'application du pooling






Etape 3
Flattening : applatissement de la matrice, on prend la matrice ligne par ligne que l'on ajoute à la suite dans un seul vecteur
Le vecteur servira de couche d'entree à un reseau de neurone
Pooling layer -----> Flattening ----> Couche d'entree ANN


Finalement : 
Input image -> f : convolution --> Convolutional layer -> f: pooling -> Pooling layer -> f: flatteninf -> Couche d'entree d'un ANN






Etape 4 : Fully connected layer (en parlant des hidden layers) pour categoriser 
On rajoute un reseau de neurone totalement connectée en plus du NN à convolution
Dans le vecteur d'entree, les features sont deja encodées et p-ê déjà visibles, le but du NN totalement connectée est d'identifier des combinaisons de features permettant de faire des prédictions

Rq : la video prend deux neurones de sortie pour identifier un chat et un chien, on avait dit avt qu'une seule sortie suffisait pour deux catégories et que ce n'est que seulement si on a 3 ou plus catégories que l'on decidait d'un neurone par catégorie
L'auteur décide de séparer pour s'exercer à interpreter plusieurs sorties

Différence par rapport au fonctionnement normal
 (ajustement du ANN en utilisant l'algo du gradient stockastique pour une fonction de coût donnée, propagation et retropropagation de l'erreur en comparant la sortie et la réalité):
On va aussi chercher à ajuster aussi les features detectors avec l'algo du gradient adapté pour ces matrices
Toute les étapes depuis le début de la convolution font partie de la retropropagation

Concernant la sortie : pour le sortie consacrée au chien
objectif : identifer quels poids sont important pour reconnaitre un chien et lesquels doivent être connectés à cette sortie
les neurones représentent plus ou moins des features (même si elles ont subi des opérations) 
pour une entrée donnée, la sortie de certains neurones de la hidden layer en fonction de l'entrée, par exple la dernière, sera plus forte
Expl: si pour entrée de chien, 3 neurones de la derniere de la hidden layer donne une sortie forte, alors p-ê celui-ci correspond à l'identification de plusieurs features de chien et on cherchera à renforcer les lien qui les lient avec le neurone de sortie "chien"
de même, on tendra à ignorer ceux qui ont des sorties faibles pour ces feature et ainsi de suite pour toutes les features que l'on peut identifier et pour chaque animal
(Attention explication simplificatrice) pour une entrée d'un chien, sachant que certaint neurones de la dernière couche de la hidden layer sont consacrées à l'identification d'un chat et d'autres d'un chien,
ceux-ci plus fortement connectées à leurs sorties respectives, "votent" càd emmettent chacun un valeur dont l'importance permettra aux neurones de sortie de calculer une proba




Softmax et entropie croiséee:
Softmax : https://fr.wikipedia.org/wiki/Fonction_softmax
en reprenant le reseau précédant, on remarque que pour une entrée donnée les proba de sortie se somment à 1 sans que les deux neurones de sortie soient connectés
En fait, ils ne se parlent pas. On utilise la fonction d'activation softmax qui donnent des probas sommant à 1. 
Si on ne l'utilisait pas, "chien" et "chat" auraient des valeurs qlq ne se sommant pas necessairement à 1.
Softmax est généralisation de la f utilisée pour la regression logistique. On l'utilise lorsqu'une entrée peut appartenir à plusieurs catégorie et que l'on utilise autant de neurones de sorte et que l'on a besoin que les probas se somment à 1 pour donner du sens à la prediction.

Entropie croisée : https://fr.wikipedia.org/wiki/Entropie_crois%C3%A9e
pour l'ANN, la fonction de cout utilisée était celle 1/2 * (Somme (y-ŷ)²)
pour le CNN, celle plus adaptée est l'entropie croisée. Le but est toujours de la minimiser.
intérêt de celle f de coût : 
Soit 2 NN: on comptabilise les prédictions pour chaque entrée


NN1  ^: prediction         Erreur de classification (Classification bonne/nb de classification):    Erreur quadratique moyenne:   Entropie croisée:
                            peu utilisé                                                             0.25                          0.71
Chien^ Chat ^ Chien Chat    1/3
0.9    0.1    1     0
0.1    0.9    0     1
0.4    0.6    0     1


NN2
Chien^ Chat ^ Chien Chat    1/3                                                                     0.38                           1.06
0.6    0.4    1     0
0.3    0.7    0     1
0.1    0.9    0     1

L'avantage de l'entropie croisée pas visible ici : supposons que tout au début de l'entrainement, on ait des valeurs de sortie/predictions minuscules  p/r à la vraie valeur.
A cause de ça, l'algo du gradient mettrant bcp de temps à ajuster les poids si l'on utilise l'erreur quad moy
Exple : supposons que la vraie "valeur" pour prédire chien est 1 et que la proba de sortie est 10^⁻6. L'erreur quadratique sera quasiment de un.
De même, pour une proba de sortie de 10^-3, l'erreur quad moy est toujours de quasiment 1 alors que la prédiction est 1000 fois plus précise qu'auparavant

Visuellement, lors de l'execution de l'algo du gradient, bien que la prédiction ait bcp changé la courbe de descente du gradient pour trouver l'erreur min est quasiment plate, ainsi la descente se fait très lentement
alors que l'erreur qui a été commise (proche de 1) est "très élevée".


Pour l'entropie croisée: si p=10^-6 l'erreur sera de 6 et si p=10^-3 l'erreur sera de 3 -> changement d'erreur plus facilement visible dc descente plus rapide

Erreur quad moy marche bien mais dans des cas où la prédiction et donc l'erreur p-ê minuscule p/r à la vraie valeur on lui préfère l'entropie croisée pour converger plus rapidement vers une solution
"""


#Etape 1

#comment structurer les données ?
#jeux d'entrainement dans 2 dossiers differents dont le nom correpond à la catégorie d'image qu'ils contiennent
#les fichiers images sont nommés avec leur catégorie suivie d'une numérotation

"""
On pourrait extraire du nom des images les catégories d'appartenance de celles-ci
On utilise un module de keras pour le faire avec au préalable ordonné les données avec la bonne structure et keras la reconnaitra
dataset -> test_set
        -> training_set -> cat -> images
                        -> dog
mis à l'echelle des données est toujours à faire !

"""


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
                                          input_shape=(64,64,3),
                                          activation="relu"))
"""
filters : integer, dimenstionalté de l'espace de sortie -> nombre de feature detectors
On prend 32 car puissance de 2 et standard pour démarrer un CNN
Si on rajoute une autre couche de Convolution, on double à chaque fois le nombre de filters

rappel : Ici, nos calculs sont effectués sur le CPU, on cherche à réduire la taille des données pour éviter des calculs trop longs
kernel = filters !

stride de 1, les feature detectors bouge d'un pas de 1 pixel

kernel_size : taille du feature detector 

Attention : When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
input_shape à specifier ici, pour image en N/B on simplement un nv de gris donc (x,y, 1 (un entier allant de 0 à 255)) car tableau en 2D et pour image en RGB (x,y,3) car tableau en 3D avec x,y taille en pixels des images
Convertir les images dans un même format 
"""


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
input_shape à changer : puisque les images en entrée de la couche sortent de la couche de convo et de pooling et que cette dernière /2 la taille (car kernel size du max pooling = 2x2)
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
classifier.add(Dense(units=1, activation="sigmoid"))
"""
Combien de neurones prendre ? Moyenne nb de neurone couche d'entree + couche de sortie ?
ici la couche de sortie contient un neurone (qui donne une proba) car on cherche à classifier des images selon 2 catégories
Ici, question épineuse car on effectue du pooling et du flattening
On prend un nombre ni trop petit ni trop grand(q de puissance) et puissance de 2 
"""


#Etape 5 : Compilation
#choix de l'algo du gradient, de la fonction de cout, de la metrics pour mesurer la performance
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
"""
ici problème de classification (et pas de regression) à deux catégorie donc usage de binary_crossentropy
"""

#Etape 6 : Fitting images on NN  + entrainement de CNN
"""
Augmentation d'image : permet de générer un plus grand nombre d'images à partir d'un set de data et d'eviter le surentrainement
modification de paramètre dans l'image (rotation, extension,miroir, brightness, colours)

on utilise un module de keras comme raccourci
sur le site keras.io on cherche la categorie Preprocessing -> Image preprocessing

situation classique de surentrainement : pas assez de données dans le jeu d'entrainement -> n'arrive pas à généraliser

usage de la methode .flow_from_directory car on a déjà les données préparés avec la structure adéquate
"""



from keras.preprocessing.image import ImageDataGenerator

"""
Augmentation des données


rescale : ajustement de l'echelle des couleurs -> normalisation des couleurs
shear_range : transvection , opération proche de la dilation comme si on tirait sur à partir d'un coin sur une image
horizontal_flip : retourner l'image horizontalement
"""
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

"""
on change l'echelle de tt les pixels pour le jeu de test
"""

test_datagen = ImageDataGenerator(rescale=1./255)

"""
creation des images
on change le chemin vers le dossier
target_size = taille des nvelles images
batch_size : pour l'algo du gradient stock, mise à jour des neurones au bout de $(batch_size)
class_mode : "type" de la prediction, ici on a deux categories donc la variable de prédiction est binary
"""

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

"""
entrainement + test des perfs
setps_per_epochs : nombre de fois que l'on va reajuster les poids
comme ici on utilise l'algo du gradient stock par lot (de 32), on va avoir (8000 images pour le training set)/32 = 250 lots contenant chacune 32 observations, on rentre 250
ici on prend 25 epochs pour avoir moins de calcul
on a 2000/32 = 62.5 arrondi à l'entier supérieur soit 63 -> validation_steps : 63

"""
classifier.fit_generator(
        training_set,
        steps_per_epoch=250,
        epochs=50,
        validation_data=test_set,
        validation_steps=63)

"""
ici ce n'est fit mais fit_generator comme méthode qui est appelée 
entraine le CNN et estime les perfs
"""

#resultat sans ajout d'une couche de convo supplémentaire : loss : 0.26 (training) acc : 0.88, val_loss : 0.72, (test) val_acc : 0.75



"""
Améliorer la perf ? Modèle plus profond ? Rajouter une couche cachée ? une couche de convo ?
Rajouter une couche dense + de convo (suivi d'une maxPooling) donne une meilleur resultat

Paramètre à faire varier en plus du nombres de neurones, epochs,batch_size : taille des images, on avait mis une target_size de 64x64 mais on pourrait l'augmenter pour avoir plus d'info à analyser mais plus de calcul


Détails : 
    Les NN fully connected prennent en input un vecteur tandis que les CNN peuvent prendre en input un tenseur
"""



#solution
"""
Changer la taille des images : plus de pixel par contre model plus long à entrainer
Plus d'époch
Ajout de deux nouvelles couches : 1 convo + 1 de maxpooling (penser à multiplier par 2 le nombre de features maps (filters dans les paramètres) ) 

Dans le NN:
2 nouvelles couches totalement co
Dropout (0.3) ajouté pour évite le surentrainement (il faut faire les essaies pour voir la difficulté à accéder à 90% de précision)


"""
