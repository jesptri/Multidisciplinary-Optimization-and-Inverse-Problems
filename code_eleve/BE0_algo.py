############################# BE0 : OPTIMISATION SANS CONTRAINTES ##################################

#######################################################################################
# Ce fichier contient la définition des algorithmes d'optimisation 
#
# A COMPLETER                                                                                                   
#   
#  Responsable: E.Flayac (emilien.flayac@isae.fr) -- 2025/2026
#  (C) Institut Supérieur de l'Aéronautique et de l'Espace (ISAE-Supaéro)
########################################################################################

import numpy as np
import scipy.optimize

global f_count, g_count, h_count
f_count=0 
g_count=0 
h_count=0

def algo_gradient_backtracking(une_f, un_x0, un_gf, un_alpha_0, un_c, un_beta, 
                               un_nit_max, un_f_count_max, une_tol_x, une_tol_f, une_tol_g):
    
    tempx = np.array(un_x0).reshape(-1, 1)
    #tempx =un_x0

    fdex = une_f(tempx)
    gfdex = un_gf(tempx)
    n = len(tempx)
    k = 0
    fin = 0
    x_k = tempx
    while fin == 0 and k < un_nit_max:
        i=0

        while fdex - un_c*un_alpha_0*un_beta**(i) *np.linalg.norm(gfdex)**2 - une_f(tempx - un_alpha_0*un_beta**i*gfdex) < 0 :
            i +=1

        alpha_k = un_alpha_0 * un_beta ** i
        f_old = fdex  

        x_k += alpha_k * (-gfdex) 
        fdex = une_f(x_k)
        gfdex = un_gf(x_k)

        # A COMPLETER
    
        if k == un_nit_max-1:
            fin = 3
    
        if np.linalg.norm(gfdex) < une_tol_g:
            fin = 1
                    
        if np.linalg.norm(alpha_k * gfdex) < une_tol_x:
            fin = 2
                
        if abs(fdex - f_old) < une_tol_f:
            fin = 4
        
        k += 1
        
    x_opt = tempx
    f_opt = une_f(tempx)
    nit = k
       
    return x_opt, f_opt, fin, nit

def algo_Newton_pure(une_f, un_x0, un_gf, une_hf, un_alpha_0, un_nit_max, un_f_count_max, une_tol_x, une_tol_f, une_tol_g):
    
    #tempx = np.array(un_x0).reshape(-1, 1)
    tempx =un_x0
    fdex = une_f(tempx)
    gfdex = un_gf(tempx)
    hfdex = une_hf(tempx)
    n = len(tempx)
    k = 0
    fin = 0
    x_k = tempx
    while fin == 0 and k < un_nit_max:
        
        d_k = np.linalg.solve(hfdex, -gfdex)
        alpha_k = 1
        x_k += alpha_k * d_k
        f_old = fdex
        fdex = une_f(x_k)
        gfdex = un_gf(x_k)
        hfdex = une_hf(x_k)
        # A COMPLETER

        if k == un_nit_max-1:
            fin = 3
    
        if np.linalg.norm(gfdex) < une_tol_g:
            fin = 1
                    
        if np.linalg.norm(alpha_k * gfdex) < une_tol_x:
            fin = 2
                
        if abs(fdex - f_old) < une_tol_f:
            fin = 4
        
        k += 1

    x_opt = tempx
    f_opt = une_f(tempx)
    nit = k

    return x_opt, f_opt, fin, nit

def algo_Newton_backtracking(une_f, un_x0, un_gf, une_hf, un_alpha_0, un_c, un_beta, un_nit_max, un_f_count_max, une_tol_x, une_tol_f, une_tol_g):
    
    tempx =un_x0
    fdex = une_f(tempx)
    gfdex = un_gf(tempx)
    hfdex = une_hf(tempx)
    n = len(tempx)
    k = 0
    fin = 0
    alpha_k = un_alpha_0 

    x_k = tempx
    while fin == 0 and k < un_nit_max:
        i=0
        
        d_k = np.linalg.solve(hfdex, -gfdex)

        while fdex + un_c*un_alpha_0*un_beta**(i) *np.dot(gfdex.T, d_k) - une_f(x_k + alpha_k*un_beta**i*d_k) < 0 :
            i +=1
            alpha_k = un_alpha_0 * un_beta ** i

        f_old = fdex  

        x_k += alpha_k * d_k
        fdex = une_f(x_k)
        gfdex = un_gf(x_k)

        if k == un_nit_max-1:
            fin = 3
    
        if np.linalg.norm(gfdex) < une_tol_g:
            fin = 1
                    
        if np.linalg.norm(alpha_k * gfdex) < une_tol_x:
            fin = 2
                
        if abs(fdex - f_old) < une_tol_f:
            fin = 4
            
        k += 1

    x_opt = tempx
    f_opt = une_f(tempx)
    nit = k

    return x_opt, f_opt, fin, nit

def algo_hybride_gradient_Newton(une_f, un_x0, un_gf, une_hf, un_alpha_0, un_c, un_beta, un_nit_max, un_f_count_max, une_tol_x, une_tol_f, une_tol_g):
    
    tempx =un_x0
    fdex = une_f(tempx)
    gfdex = un_gf(tempx)
    hfdex = une_hf(tempx)
    n = len(tempx)
    k = 0
    fin = 0
    alpha_k = un_alpha_0 

    x_k = tempx
    while fin == 0 and k < un_nit_max:
        i=0
        
        d_k_barre = np.linalg.solve(hfdex, -gfdex)

        if np.dot(un_gf(x_k).T,d_k_barre)<0:
            d_k = d_k_barre
        else:
            d_k = -d_k_barre

        while fdex + un_c*un_alpha_0*un_beta**(i) *np.dot(gfdex.T, d_k) - une_f(x_k + alpha_k*un_beta**i*d_k) < 0 :
            i +=1
            alpha_k = un_alpha_0 * un_beta ** i

        f_old = fdex  

        x_k += alpha_k * d_k
        fdex = une_f(x_k)
        gfdex = un_gf(x_k)
        
        # A COMPLETER

        if k == un_nit_max-1:
            fin = 3
    
        # if #A COMPLETER
        #     fin = 2
            
        # if #A COMPLETER
        #     fin = 1
        
        # if #A COMPLETER
        #     fin = 4
        
        k += 1

    x_opt = tempx
    f_opt = une_f(tempx)
    nit = k

    return x_opt, f_opt, fin, nit

def algo_QuasiNewton_backtracking(une_f, un_x0, un_gf, un_B0, un_alpha_0, un_c, un_beta, un_nit_max, un_f_count_max, une_tol_x, une_tol_f, une_tol_g):
    # Global variables
    
    global f_count, g_count, h_count
    n=len(un_x0)
    tempx = np.array(un_x0).reshape(-1, 1)
    fdex = une_f(tempx)
    gfdex = un_gf(tempx)
    B_k = un_B0
    n = len(tempx)
    k = 0
    fin = 0
    x_k = un_x0
    alpha_k = un_alpha_0
    while fin == 0 and k < un_nit_max:

        x_k_old = x_k 

        d_k = -B_k @ un_gf(x_k)
        i = 0
        while fdex + un_c*un_alpha_0*un_beta**(i) *np.dot(gfdex.T, d_k) - une_f(x_k + alpha_k*un_beta**i*d_k) < 0 :
            i +=1
            alpha_k = un_alpha_0 * un_beta ** i
        
        # A COMPLETER

        x_k += alpha_k * d_k
        y_k = un_gf(x_k) - un_gf(x_k_old)
        z_k = x_k - x_k_old
        rho_k = 1/(np.dot(y_k.T,z_k))
        B_k = (np.eye(n) - rho_k * z_k @ y_k.T) @ B_k @ (np.eye(n) - rho_k * y_k @ z_k.T) + rho_k * z_k @ z_k.T

        f_old = fdex
        fdex = une_f(x_k)
        gfdex = un_gf(x_k)

        if k == un_nit_max-1:
            fin = 3
    
        if np.linalg.norm(gfdex) < une_tol_g:
            fin = 1
                    
        if np.linalg.norm(alpha_k * gfdex) < une_tol_x:
            fin = 2
                
        if abs(fdex - f_old) < une_tol_f:
            fin = 4
        
        k += 1

    x_opt = tempx
    f_opt = une_f(tempx)
    nit = k

    return x_opt, f_opt, fin, nit

def algo_accelerated_gradient(une_f, un_x0, un_gf, un_alpha_0, un_c, un_beta, 
                               un_nit_max, un_f_count_max, une_tol_x, une_tol_f, une_tol_g):
    
    tempx = np.array(un_x0).reshape(-1, 1)
    tempy = np.array(un_x0).reshape(-1, 1)
    #tempx =un_x0

    fdex = une_f(tempx)
    gfdex = un_gf(tempx)
    gfdey = un_gf(tempy)
    n = len(tempx)
    k = 0
    fin = 0

    while fin == 0 and k < un_nit_max:
       
        # A COMPLETER

        if k == un_nit_max-1:
            fin = 3
    
        # if #A COMPLETER
        #     fin = 2
            
        # if #A COMPLETER
        #     fin = 1
        
        # if #A COMPLETER
        #     fin = 4
        
        k += 1
        
    x_opt = tempx
    f_opt = fdex
    nit = k
       
    return x_opt, f_opt, fin, nit