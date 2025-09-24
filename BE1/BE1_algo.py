#######################################################################################
# Ce fichier contient la d�finition des algorithmes d'optimisation                    #
#######################################################################################



#######################################################################################
#                                   PARAMETRES EN ENTREE                              #
#######################################################################################

          #   une_f                 une fonction dont on cherche un minimum    

          #   des_c                 les contraintes des probleme               

          #   un_x0                 un incontournable point initial  
          #  un_lambda0             une valeur intiale pour les multiplicateurs de Lagrange.
           

          #   un_gf                 une fonction qui code le gradient de une_f 


          #  jac_des_c              une fonction qui code la jacobienne de des_c 

          #   un_hf                 une fonction qui code le Hessien de  une_f
          #   h_des_c               une fonction qui code les Hessiens de des_c 

          #   un_nit_max            nombre maximum d'iterations autorisees     
          #                         risees                                     
          #   une_tol_x             seuil de stationnarite des x_k             
          #   une_tol_f             seuil de stationnarite des f_k             
          #   une_tol_g             seuil validant EULER    en x_k             
          #                          le sous-probleme:
          #                          (1 exact, 2 pas de Cauchy, 3 CG tronqué)   
          #   une_tol_h             pas de discretisation des derivees 

#######################################################################################
#                                   PARAMETRES EN SORTIE                              #
#######################################################################################           

          #   x_opt                 la solution proposee par trust_region      
          #   f_opt                 une_f (x_opt, varargin{:})                 
          #   g_opt                 un_gf (x_opt, varargin{:})                 
          #   fin                   la cause de l'arret de l'algorithme        
          #   nit                   le nombre iterations de trust_region       
          #   f_count               le nombre d'evaluations de une_f           
          #   g_count               le nombre d'evaluations de un_gf

#  Responsable: E.Flayac (emilien.flayac@isae.fr) -- 2023/2024
#  (C) Institut Superieur de l'Aeronautique et de l'Espace (ISAE-Supaero)
########################################################################################



global f_count, g_count, h_count, c_count, jc_count, hc_count

f_count = 0
g_count = 0
h_count = 0
c_count = 0
jc_count = 0
hc_count = 0






import numpy as np

def algo_SQP(une_f, des_c, un_x0, un_lambda0, un_gf, jac_des_c, un_hf, h_des_c, un_nit_max, une_tol_x, une_tol_f, une_tol_g):

    #tempx et templambda sont des vecteurs colonne
    tempx = np.array(un_x0, dtype=float).reshape(-1, 1)
    templambda = np.array(un_lambda0, dtype=float).reshape(-1, 1)
    
    # tempx=np.array(un_x0)
    # templambda = np.array(un_lambda0)   
    n = len(tempx)
    p = len(templambda)
    


    k = 0
    fin = 0
    # print(templambda)
    x_k = tempx
    lambda_k = templambda

    while fin == 0 and k < un_nit_max:

        #Evaluation du Lagrangien
        fdex = une_f(x_k)
        cdex = des_c(x_k)
        ldex = fdex + np.dot(lambda_k, cdex)
        #Evaluation du gradient
        gfdex = un_gf(x_k)
        jcdex = jac_des_c(x_k).reshape(1,-1) 
        grad_lagrangien = gfdex + np.dot(jcdex.T, lambda_k)
   


        #évaluation de la hessienne
        hfdex = un_hf(tempx)
        hcdex = h_des_c(tempx)
        hess_lagrangien = hfdex + lambda_k.T*hcdex
           
        A = hess_lagrangien
        B = jcdex.T
        C = jcdex
        D = np.zeros((p, p))
        M = np.block([[A, B],[C, D]])

        E = -gfdex
        F = -cdex
        Y = np.block([[E],[F]])

        #A COMPLETER

        solution = np.linalg.solve(M,Y)

        d_k = solution[:n].reshape(-1, 1)
        lambda_k = solution[n:].reshape(-1, 1)

        x_old = x_k.copy()
        x_k += d_k
        
        if k == un_nit_max-1:
            fin = 1

        if np.linalg.norm(d_k) < une_tol_x:
            fin = 2

        if abs(une_f(x_k) - une_f(x_old)) < une_tol_f:
            fin = 3

        # if np.linalg.norm(grad_lagrangien) < une_tol_g:
        #     fin = 4

        k=k+1    
            
    x_opt = tempx
    f_opt = fdex
    nit = k

    return x_opt, f_opt, fin, nit


# def algo_SQP(une_f, des_c, un_x0, un_lambda0, un_gf, jac_des_c, un_hf, h_des_c, un_nit_max, une_tol_x, une_tol_f, une_tol_g):

#     #tempx et templambda sont des vecteurs colonne
#     tempx = np.array(un_x0).reshape(-1, 1)
#     templambda = np.array(un_lambda0).reshape(-1, 1)
    
    
#     # tempx=np.array(un_x0)
#     # templambda = np.array(un_lambda0)   
#     n = len(tempx)
#     p = len(templambda)
    
    
#     #Evaluation du Lagrangien
#     fdex = une_f(tempx)
#     cdex = des_c(tempx)

#     #Evaluation du gradient du Lagrangien
#     gfdex = un_gf(tempx)
#     jcdex = jac_des_c(tempx).reshape(1,-1)
    
#     #Evaluation de la hessienne du Lagrangien
#     hfdex = un_hf(tempx)
#     hcdex = h_des_c(tempx)

#     k = 0
#     fin = 0
#     while fin == 0 and k < un_nit_max:
        
#         #A COMPLETER
        
#         if k == un_nit_max-1:
#             fin = 3
       
#         k=k+1    
            
#     x_opt = tempx
#     f_opt = fdex
#     nit = k

#     return x_opt, f_opt, fin, nit

def algo_SQP_BFGS(une_f, des_c, un_x0, un_lambda0, un_gf, jac_des_c, H0, un_nit_max, une_tol_x, une_tol_f, une_tol_g):

    tempx = np.array(un_x0, dtype=float).reshape(-1, 1)
    templambda = np.array(un_lambda0, dtype=float).reshape(-1, 1)

    n = len(tempx)
    p = len(templambda)
    
    #Evaluation du Lagrangien
    fdex = une_f(tempx)
    cdex = des_c(tempx)

    #Evaluation du gradient du Lagrangien
    gfdex = un_gf(tempx)
    jcdex = jac_des_c(tempx).reshape(1,-1)
    print(cdex)
    print(jcdex)
    #Approximation de la hessienne du Lagrangien
    
    H_old=H0
    H_k=H0
    grad_lagrangien_k = gfdex + np.dot(jcdex.T, templambda)




    k = 0
    fin = 0
    # print(templambda)
    x_k = tempx
    lambda_k = templambda   

    while fin == 0 and k < un_nit_max:
           
        A = H_k
        B = jcdex.T
        C = jcdex
        D = np.zeros((p, p))
        M = np.block([[A, B],[C, D]])

        E = -gfdex
        F = -cdex
        Y = np.block([[E],[F]])

        #A COMPLETER
        

        solution = np.linalg.solve(M,Y)

        d_k = solution[:n].reshape(-1, 1)

        lambda_k = solution[n:].reshape(-1, 1)

        x_old = x_k.copy()


        x_k += d_k

        #Evaluation du Lagrangien
        fdex = une_f(x_k)
        cdex = des_c(x_k)
        #Evaluation du gradient
        gfdex = un_gf(x_k)
        jcdex = jac_des_c(x_k).reshape(1,-1) 
        grad_lagrangien_old = grad_lagrangien_k.copy()
        grad_lagrangien_k = gfdex + np.dot(jcdex.T, lambda_k)

        
        #Evaluation de Hk

        s_k = x_k - x_old
        y_k = grad_lagrangien_k - grad_lagrangien_old

        if y_k.T @ s_k > 0 :

            H_old = H_k.copy()
            f1 = np.dot(y_k,y_k.T)/np.dot(y_k.T, s_k)
            f2 = (H_old @ s_k @ s_k.T @ H_old) / (s_k.T @ H_old @ s_k)
            H_k = H_old + f1 - f2
        
        if k == un_nit_max-1:
            fin = 1

        # if np.linalg.norm(d_k) < une_tol_x:
        #     fin = 2

        # if abs(une_f(x_k) - une_f(x_old)) < une_tol_f:
        #     fin = 3

        if np.linalg.norm(grad_lagrangien_k) < une_tol_g:
            fin = 4
       
        k=k+1      
            
    x_opt = tempx
    f_opt = fdex
    nit = k

    return x_opt, f_opt, fin, nit


    ######## ATTENTION ###################
    ### Les deux suites de commande suivantes ne donnent pas le meme resultat:
        
    #####cas 1######                          
    # a=np.array([1,2])
    # old_a=a
    # a[0]=3
    # print(a)
    # print(old_a)

    #####cas 2######
    # a=np.array([1,2])
    # old_a=a
    # a=np.array([3,2])
    # print(a)
    # print(old_a)

    ### Afin d'obtenir le meme resultat que le cas 2 en utilisant 
    ### une affectation comme dans le cas 1, il faut utiliser la methode copy()
    
    #####cas 3######                          
    # a=np.array([1,2])
    # old_a=a.copy()
    # a[0]=3
    # print(a)
    # print(old_a)
    


def algo_SQP_sans_derivee(une_f, des_c, un_x0, un_lambda0, H0, une_tol_h, un_nit_max, une_tol_x, une_tol_f, une_tol_g):

    
    tempx = np.array(un_x0, dtype=float).reshape(-1, 1)
    templambda = np.array(un_lambda0, dtype=float).reshape(-1, 1)

    n = len(tempx)
    p = len(templambda)
    h = 1e-5
    #Evaluation du Lagrangien
    fdex = une_f(tempx)
    cdex = des_c(tempx)

    #Approximation du gradient du Lagrangien
    gfdex=np.zeros((n,1))

    for i in range(n):
        gfdex[i,0] = (une_f(tempx + h*np.eye(n)[:, i]) - une_f(tempx))/h
    jcdex = np.zeros((1,n))
    for i in range(n):
        jcdex[0,i] = (des_c(tempx + h*np.eye(n)[:, i]) - des_c(tempx))/h

    #Approximation de la hessienne du Lagrangien
    
    H_old=H0
    H_k=H0
    grad_lagrangien_k = gfdex + np.dot(jcdex.T, templambda)




    k = 0
    fin = 0
    # print(templambda)
    x_k = tempx
    lambda_k = templambda   

    while fin == 0 and k < un_nit_max:
           
        A = H_k
        B = jcdex.T
        C = jcdex
        D = np.zeros((p, p))
        M = np.block([[A, B],[C, D]])

        E = -gfdex
        F = -cdex
        Y = np.block([[E],[F]])

        #A COMPLETER
        

        solution = np.linalg.solve(M,Y)

        d_k = solution[:n].reshape(-1, 1)

        lambda_k = solution[n:].reshape(-1, 1)

        x_old = x_k.copy()


        x_k += d_k

        #Evaluation du Lagrangien
        fdex = une_f(x_k)
        cdex = des_c(x_k)
            #Evaluation du gradient
        for i in range(n):
            gfdex[i,0] = (une_f(x_k + h*np.eye(n)[:, i]) - une_f(x_k))/h

        for i in range(n):
            jcdex[0,i] = (des_c(x_k + h*np.eye(n)[:, i]) - des_c(x_k))/h

        #jcdex = jac_des_c(x_k).reshape(1,-1) 
        grad_lagrangien_old = grad_lagrangien_k.copy()
        grad_lagrangien_k = gfdex + np.dot(jcdex.T, lambda_k)

        
        #Evaluation de Hk

        s_k = x_k - x_old
        y_k = grad_lagrangien_k - grad_lagrangien_old

        if y_k.T @ s_k > 0 :

            H_old = H_k.copy()
            f1 = np.dot(y_k,y_k.T)/np.dot(y_k.T, s_k)
            f2 = (H_old @ s_k @ s_k.T @ H_old) / (s_k.T @ H_old @ s_k)
            H_k = H_old + f1 - f2
        
        if k == un_nit_max-1:
            fin = 1

        # if np.linalg.norm(d_k) < une_tol_x:
        #     fin = 2

        # if abs(une_f(x_k) - une_f(x_old)) < une_tol_f:
        #     fin = 3

        # if np.linalg.norm(grad_lagrangien_k) < une_tol_g:
        #     fin = 4
       
        k=k+1      
            
    x_opt = tempx
    f_opt = fdex
    nit = k


    return x_opt, f_opt, fin, nit
