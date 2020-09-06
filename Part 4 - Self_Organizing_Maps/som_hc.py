# -*- coding: utf-8 -*-

"""
comment les cartes auto adaptatives fonctionnent
k means algo ?
commeent les som apprennent ?
"""

#Cartes auto adaptatives
#Detection de fraudes de banque (plutôt identifier un groupe de personnes suspectes)

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importation des données
dataset = pd.read_csv("Credit_Card_Applications.csv")

"""
difficulté : jeu de données anonymisées (pas de nom attribués aux colonnes)
colonnes = réponses, ligne = client
objectifs : segmenter les clients (regrouper les clients similaires entre eux)

couche d'entrée : clients (1 neurone par varibles)
sortie :"grille" de neurones représentant la carte finale
entre deux : chaque neurones de sortie à un vecteur de poids qui correspond à des coordonnées dans l'espace d'entrée
ici de dim 15 (car 15 variables) (14 variables + customer id ( on exclut la var classe )

1 epoch : pour un client donnée recherche du neurone le plus proche + tirage de celui ci plus de son voisinnage vers ce point
"""

"""
comment detecter la fraude : la fraude sera identifiable par un profil atypique càd à un neurone très éloigné de ts les autres
usage de mean inter neurons distance , pour chaque neurone de la carte de sortie on calcul sa distance avec ceux de son voisinnage puis moyenne
si neurone éloigné de son voisinnage -> d moyenne élevée

après avoir identifié ce neurone éloignée identifié comme fraude -> function inverse pour identifier les neurones d'entrée (càd les clients) liés à celui-ci
"""


"""
partage du jeu de données en 2
var : id à c14, class-> indique si oui ou non la demande d'adhésion à la banque d'un client est acceptée
le client p-ê refusé sans être un fraudeur ou accepté en étant un

intére : on peut s'intéresser à des cas plus particulier (en l'occurence le deuxième cas)
"""


x = dataset.iloc[:,:-1].values #iloc[ tts lignes, ttles colonnes jusqu'à indice -1 (dernière) non comprise ?]
y = dataset.iloc[:,-1].values #y n'est pas un vecteur à prédire comme ds l'apprentissage supervisé


#changement d'échelle des var (accelère les calculs)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x = sc.fit_transform(x)


#Entrainement du SOM
#il n'existe pas d'auto implimentation des SOM dans keras ou sklearn
#bonne adaptation sur le site MiniSum
from minisom import MiniSom
som = MiniSom(10,10, input_len=15) # x et y = dims de la carte auto adaptative, (+ elle est gde + elle est précise) -> juste milieu (comme pas énormément de ligne, 10x10)
som.random_weights_init(x)
som.train_random(x,num_iteration=100) #entrainement rapide ici car pas bcp de données

#visualisation des résultats
from pylab import bone, pcolor,plot, colorbar, show #outils, graph -> auto (pour suivre ttes les étapes)
bone() # init graph
pcolor(som.distance_map().T) # transform val en couleur, T (rotation de 90 deg)
colorbar()

#cas particulier : client accepté fraudeur
markers = ["o", "s"] #marqueur : cercle = o, carrée : square s
colors = ["r", "g"] # r = refusé, green = accepté
for i, ligne in enumerate(x): #i = index of line, ligne bah la ligne
    #mise en lumière du neurone "gagnant" de la ligne "ligne" (càd celui le (vecteur) client le plus proche du (vecteur) neurone donné)
    w = som.winner(ligne)
    plot(w[0]+ 0.5, w[1] + 0.5,
        markers[y[i]],
        markeredgecolor = colors[y[i]],
        markerfacecolor = "None",
        markersize=10,
        markeredgewidth=2) #+0.5 pour centrer le marker
show()


#fonction inverse : déterminer les clients pour un grp de neurones donnés
#fonction inverse n'existe pas mais on peut utiliser un dico pour stocker les listes de clients liés au groupe de neurones d'intérêt
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(6,4)],mappings[(1,0)], mappings[(1,4)]),
                        axis=0)
#inversion des échelles
frauds = sc.inverse_transform(frauds)

