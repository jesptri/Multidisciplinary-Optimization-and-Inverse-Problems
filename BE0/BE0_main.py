############################# BE0 : OPTIMISATION SANS CONTRAINTES ##################################





#######################################################################
# Ce fichier permet de choisir l'algorithme à éxecuter et le critère à minimiser 
# en utilisant la fonction BE0_main à laquelle il faut passer les arguments "algo" et "prob" 
# selon la correspondance suivante :   
#                                                                                                       
#     algo: 1, 2, 3, 4 ou 5 pour choisir l'algorithme.
#        algo=1 Gradient avec Recherche lineaire
#        algo=2 Newton Basique
#        algo=3 Newton avec Recherche lineaire
#        algo=4 Gradient-Newton avec Recherche lineaire
#        algo=5 Quasi-Newton avec Recherche lineaire
#     prob: 1, 2 ou 3.
#        prob=1 choix de la fonction f1 "bad"
#        prob=2 choix de la fonction f1 "hard"
#        prob=3 choix de la fonction f1 "rosenbrock"

#  Responsable: E.Flayac (emilien.flayac@isae.fr) -- 2024/2025
#  (C) Institut Supérieur de l'Aéronautique et de l'Espace (ISAE-Supaéro)
#########################################################################



import time

import numpy as np

from BE0_noyau import BE0_main




#**************************
# GLOBALES EN MISE A JOUR *
#**************************


global f_count                  
global g_count                   
global h_count
                    
f_count = 0
g_count = 0
h_count = 0



############# Exemple utilisation#################################
BE0_main(1, 1)
# BE0_main(1, 2)
# BE0_main(1, 3)

