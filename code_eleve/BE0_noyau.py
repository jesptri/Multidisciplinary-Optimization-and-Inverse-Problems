############################# BE0 : OPTIMISATION SANS CONTRAINTES ##################################


#######################################################################################
# Ce fichier organise l'éxecution des algorithmes et 
#contient la définition des critères à minimiser ainsi que leurs dérivées.   
#   
# A NE PAS MODIFIER                                                                                                    
#   
#  Responsable: E.Flayac (emilien.flayac@isae.fr) -- 2024/2025
#  (C) Institut Supérieur de l'Aéronautique et de l'Espace (ISAE-Supaéro)
########################################################################################




import time

import numpy as np


from BE0_algo import algo_gradient_backtracking, algo_Newton_pure, algo_Newton_backtracking, algo_hybride_gradient_Newton, algo_QuasiNewton_backtracking, algo_accelerated_gradient






#**************************
# GLOBALES EN MISE A JOUR *
#**************************


global f_count                  
global g_count                   
global h_count
                    
# f_count = 0
# g_count = 0
# h_count = 0





# Function definitions

def f_good(x):
    global f_count

    fdex = 0.5 * (x[0])**2 + 0.5 * (x[1])**2 + 0.25 * (x[0]**2)**2 + 0.25 * (x[1]**2)**2
    f_count += 1

    return fdex

def f_bad(x):
    global f_count

    fdex = 100 * x[0]**4 + 0.01 * x[1]**4
    f_count += 1

    return fdex



def f_hard(x):
    global f_count

    fdex = (1 + x[0]**2)**0.5 + (1 + x[1]**2)**0.5
    f_count += 1

    return fdex


def f_rosenbrock(x):
    global f_count

    fdex = 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    f_count += 1

    return fdex



# Gradient  definitions


def g_good(x):
    global g_count


    gfdex = np.array([x[0]+x[0]**3, x[1]+x[1]**3])
    g_count += 1

    return gfdex

def g_bad(x):
    global g_count

    gfdex = np.array([400 * x[0]**3, 0.04 * x[1]**3])
    g_count += 1

    return gfdex

def g_hard(x):
    global g_count

    gfdex = np.array([x[0]/((1+x[0]**2)**0.5), x[1]/((1+x[1]**2)**0.5)])
    g_count += 1

    return gfdex


def g_rosenbrock(x):
    global g_count

    gfdex = np.array([-400 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])
    g_count += 1

    return gfdex

# Hessian definitions

def h_good(x):
    global h_count

    hfdex = np.array([[1 + 3 * x[0]**2, 0], [0, 1 + 3 * x[1]**2]])
    h_count += 1

    return hfdex


def h_bad(x):
    global h_count

    hfdex = np.array([[1200 * x[0]**2, 0], [0, 0.12 * x[1]**2]])
    h_count += 1

    return hfdex

def h_hard(x):
    global h_count

    hfdex = np.array([[(1/np.sqrt(1+x[0]**2))**3, 0], [0, (1/np.sqrt(1+x[1]**2))**3]])
    h_count += 1

    return hfdex


def h_rosenbrock(x):
    global h_count

    hfdex = np.array([[1200*x[0]**2-400*x[1]+2, -400*x[0]], [-400*x[0], 200]])
    h_count += 1

    return hfdex














# Function definition
def BE0_main(algo, prob):
    global f_count                  
    global g_count                   
    global h_count  
    
     # Print header
    ligne_tiret = '-' * 105
    lentete = 'DEPART          METHODE           PROBLEME    FIN   F_COUNT  G_COUNT  H_COUNT  NITER      F_OPT   '
    
    
    # Constants
    nom_algo = ['Gradient_Backtraking', '     Newton_Pure    ', ' Newton_Backtraking ', ' Gradient_Newton_Bkt', '  BFGS_Backtraking  ', 'Accelerated_Gradient']
    nom_prob = ['   Good   ','    Bad   ', '   Hard   ', 'Rosenbrock']
    nom_point = [' (1, 1) ', ' (2, 2) ', '(10, 10)', ' (2, 5) ']

   

    # Initialisaiton des vecteures
    top_on = [0] * 4
    x_opt = [[0] * 2 for _ in range(4)]
    f_opt = [0] * 4
    conv = [0] * 4
    ite = [0] * 4
    temps = [0] * 4
    x_optimale = [[[0] * 2 for _ in range(4)] for _ in range(6)]
    t_cpu_time = [[0] * 4 for _ in range(6)]
    t_fin = [[0] * 4 for _ in range(6)]
    t_f_count = [[0] * 4 for _ in range(6)]
    t_g_count = [[0] * 4 for _ in range(6)]
    t_h_count = [[0] * 4 for _ in range(6)]
    t_nit = [[0] * 4 for _ in range(6)]
    t_f_opt = [[0] * 4 for _ in range(6)]



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#      ETAPE DE RESOLUTION DU PROBLEME D'OPTIMISATION                     #
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#

# en opti non-lineaire il faut un
# point de recherche initial ! x0

    x = np.array([
        [1., 1.],
            [2., 2.],
            [10., 10.],
            [2., 5.]
            ])

# Choix de probleme de minimisation (f1, f2 ou f3)

    if prob == 1:
        f_fun = f_good
        g_fun = g_good
        h_fun = h_good
    elif prob == 2:
        f_fun = f_bad
        g_fun = g_bad
        h_fun = h_bad
    elif prob == 3:
        f_fun = f_hard
        g_fun = g_hard
        h_fun = h_hard
    else:
        f_fun = f_rosenbrock
        g_fun = g_rosenbrock
        h_fun = h_rosenbrock

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#     Algorithme de Gradient avec backtraking 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    if algo == 1:
        
        f_count = 0
        g_count = 0
        h_count = 0

        for i in range(4):
            top_on[i] = time.process_time()
            x_opt[i], f_opt[i], conv[i], ite[i] = algo_gradient_backtracking(
                f_fun,
                x[i],
                g_fun,
                1,
                0.5,
                0.5,
                100000,
                100000,
                1e-5,
                1e-5,
                1e-5
                )
            # Store the results for display
            temps[i] = time.process_time() - top_on[i]
            x_optimale[i][algo-1] = x_opt[i]
            t_cpu_time[algo-1][i] = temps[i]
            t_fin[algo-1][i] = conv[i]
            t_f_count[algo-1][i] = f_count
            t_g_count[algo-1][i] = g_count
            t_h_count[algo-1][i] = h_count
            t_nit[algo-1][i] = ite[i]
            t_f_opt[algo-1][i] = f_opt[i]
            
            f_count = 0
            g_count = 0
            h_count = 0


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#     Algorithme de Newton Basique 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    if algo == 2:
        
        f_count = 0
        g_count = 0
        h_count = 0

        for i in range(4):
            top_on[i] = time.process_time()
            x_opt[i], f_opt[i], conv[i], ite[i] = algo_Newton_pure(
                f_fun,
                x[i],
                g_fun,
                h_fun,
                1,
                1000,
                1000,
                1e-5,
                1e-5,
                1e-5
                )
            # Store the results for display
            temps[i] = time.process_time() - top_on[i]
            x_optimale[i][algo-1] = x_opt[i]
            t_cpu_time[algo-1][i] = temps[i]
            t_fin[algo-1][i] = conv[i]
            t_f_count[algo-1][i] = f_count
            t_g_count[algo-1][i] = g_count
            t_h_count[algo-1][i] = h_count
            t_nit[algo-1][i] = ite[i]
            t_f_opt[algo-1][i] = f_opt[i]
            
            f_count = 0
            g_count = 0
            h_count = 0


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#     Algorithme de Newton avec bactracking
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    if algo == 3:
        
        f_count = 0
        g_count = 0
        h_count = 0

        for i in range(4):
            top_on[i] = time.process_time()
            x_opt[i], f_opt[i], conv[i], ite[i] = algo_Newton_backtracking(
                f_fun,
                x[i],
                g_fun,
                h_fun,
                1,
                0.5,
                0.5,
                1000,
                1000,
                1e-5,
                1e-5,
                1e-5
                )
            # Store the results for display
            temps[i] = time.process_time() - top_on[i]
            x_optimale[i][algo-1] = x_opt[i]
            t_cpu_time[algo-1][i] = temps[i]
            t_fin[algo-1][i] = conv[i]
            t_f_count[algo-1][i] = f_count
            t_g_count[algo-1][i] = g_count
            t_h_count[algo-1][i] = h_count
            t_nit[algo-1][i] = ite[i]
            t_f_opt[algo-1][i] = f_opt[i]
            
            
            f_count = 0
            g_count = 0
            h_count = 0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#     Algorithme de Gradient/Newton avec bactracking
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    if algo == 4:
        
        f_count = 0
        g_count = 0
        h_count = 0

        for i in range(4):
            top_on[i] = time.process_time()
            x_opt[i], f_opt[i], conv[i], ite[i] = algo_hybride_gradient_Newton(
                f_fun,
                x[i],
                g_fun,
                h_fun,
                1,
                0.5,
                0.5,
                1000,
                1000,
                1e-5,
                1e-5,
                1e-5
                )
            # Store the results for display
            temps[i] = time.process_time() - top_on[i]
            x_optimale[i][algo-1] = x_opt[i]
            t_cpu_time[algo-1][i] = temps[i]
            t_fin[algo-1][i] = conv[i]
            t_f_count[algo-1][i] = f_count
            t_g_count[algo-1][i] = g_count
            t_h_count[algo-1][i] = h_count
            t_nit[algo-1][i] = ite[i]
            t_f_opt[algo-1][i] = f_opt[i]
            
            
            f_count = 0
            g_count = 0
            h_count = 0
            
            
            
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#     Algorithme de Quasi-Newton avec bactracking
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    if algo == 5:
        
        f_count = 0
        g_count = 0
        h_count = 0

        for i in range(4):
            top_on[i] = time.process_time()
            x_opt[i], f_opt[i], conv[i], ite[i] =  algo_QuasiNewton_backtracking(
                f_fun,
                x[i],
                g_fun,
                np.eye(2,2),
                1,
                0.5,
                0.5,
                1000,
                1000,
                1e-5,
                1e-5,
                1e-5
                )
            # Store the results for display
            temps[i] = time.process_time() - top_on[i]
            x_optimale[algo-1][i] = x_opt[i]
            t_cpu_time[algo-1][i] = temps[i]
            t_fin[algo-1][i] = conv[i]
            t_f_count[algo-1][i] = f_count
            t_g_count[algo-1][i] = g_count
            t_h_count[algo-1][i] = h_count
            t_nit[algo-1][i] = ite[i]
            t_f_opt[algo-1][i] = f_opt[i] 
            
            f_count = 0
            g_count = 0
            h_count = 0






#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# ON AFFICHE TOUS LES RESULTATS  DE TOUS LES TESTS SOUS FORME D'UN TABLEAU#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    print('')
    print(ligne_tiret)
    print(f'| {lentete} |')
    print(ligne_tiret)
    #les_formats=['|%s| %s |%s |  %3.0f  |   %6.0f    |     %5.0f  |   %5.0f  |' ...
   # '    %5.0f   |%10.5g  | '                      ];
    # Iterate through points of departure
    for i in range(4):
        #x_opt, f_opt, fin, nit, f_count, g_count, h_count = x_optimale[:, i, algo], t_f_opt[algo, i], t_fin[algo, i], t_nit[algo, i], t_f_count[algo, i], t_g_count[algo, i], t_h_count[algo, i]
        x_opt = x_optimale[:][algo-1][i]
        f_opt = float(t_f_opt[algo-1][i])
        fin = t_fin[algo-1][i]
        nit = t_nit[algo-1][i]
        f_count = t_f_count[algo-1][i]
        g_count = t_g_count[algo-1][i]
        h_count = t_h_count[algo-1][i]
        print(f'| {nom_point[i]} | {nom_algo[algo-1]} | {nom_prob[prob-1]} | {fin:2d} | {f_count:7d} | {g_count:6d} | {h_count:6d} | {nit:5d} |   {f_opt:6.5f} |')

    print(ligne_tiret)


