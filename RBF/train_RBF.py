import numpy as np
from sklearn.cluster import KMeans
def radial_n_var(x, m, invK):
    return np.exp(-0.5* (np.dot(np.transpose(x - m), np.dot(invK, (x - m))))) # Formula PDF de n variaveis.


def trainRBF(xin, yin, p : int, r):
    N = xin.shape[0] # Number of samples
    n = xin.shape[1] # Input dimension
    xin = np.array(xin)
    yin = np.array(yin)
    
    x_clusters = KMeans(n_clusters = p, random_state = 0, n_init = 'auto').fit(xin) # Relizando o cluster sobre os dados de entrada.
    
    m = np.array(x_clusters.cluster_centers_) # Pegando os centros de cada cluster.
    covi = r*np.eye(n)
    inv_covi = (1 / r)*np.eye(n)# Calculando a matriz de covariancia inversa.

    # Para cada dado de entrada : 
    # 1- Aplicar a radial_n_var do dado de entrada para cada uma das p centróides, e por fim, encontrar toda a matriz H.
    # 2- Encontrar a matriz H para cada entrada em relação a cada centróide
    # 3- Lembrar que H[j, i] representa a distancia de cada ponto J até um centroide i.
    H = np.zeros((N,p))
    for j in range(N):
        for i in range(p):
            mi = m[i,]
            H[j, i] = radial_n_var(xin[j, ], mi, inv_covi)
            
    # Com a matriz H, a solução será obtida através da pseudoinversa.
    ones = np.ones((H.shape[0], 1))
    Haug = np.concatenate((H, ones), axis = 1)
    W = np.linalg.pinv(Haug, yin)
    return [m, covi, r, W, H]





def y_RBF(xin, modRBF):
    
    N = xin.shape[0] # Number of samples
    n = xin.shape[1] # Input dimension
    
    m = np.array(modRBF[0]) # Matriz das medias
    covi = modRBF[1] # Matriz de covariâncias
    inv_cov = (1 / modRBF[2])*np.eye(n) # Inversa da Matriz de Covariância
    Htr = modRBF[4] # Projeção camada intermediária conjunto de treinamento.
    p = Htr.shape[1] # Número de funções radiais (centróides ou neurônios da camada intermediária.)
    W = modRBF[3]
    x = np.array(xin)
    
    H = np.zeros((N,p))
    for j in range(N):
        for i in range(p):
            mi = m[i,]
            H[j, i] = radial_n_var(xin[j, ], mi, inv_cov)

    ones = np.ones((H.shape[0], 1))
    Haug = np.concatenate((H, ones), axis = 1)
    Yhat = np.dot(Haug, W)# Saída da rede.
    
    return Yhat
    
    