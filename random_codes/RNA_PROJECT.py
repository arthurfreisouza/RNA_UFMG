import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rpy2.robjects as ro
from functools import partial
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.ipython import html
html.html_rdataframe = partial(html.html_rdataframe, table_class = "docutils")
%load_ext rpy2.ipython


%%R
library(mlbench)
datasetxor <- mlbench.xor(300)
XR_datasetxor <- datasetxor$x
LABELSR_datasetxor<- datasetxor$classes



from rpy2.robjects import numpy2ri
numpy2ri.activate()
x_df_xor = np.array(ro.r['XR_datasetxor'])
labels_df_xor = np.array(ro.r['LABELSR_datasetxor'])
labels_df_xor[labels_df_xor == 1] = -1
labels_df_xor[labels_df_xor == 2] = 1


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df_xor, labels_df_xor, random_state = 0, train_size = 0.7)








def train_ELM_DropOut(xin : np.ndarray, yin : np.ndarray, p : int, keep_rate : float, control : bool) -> list:
    np.set_printoptions(precision=10)
    np.random.seed(np.random.randint(0, 10000))
    
    
    n = xin.shape[1] # Pegando o número de valores de cada entrada.

    # Z[n ou n + 1, p]
    if control == True:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range((n + 1) * p)]).reshape(n + 1, p)
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin,ones), axis = 1)
    else:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range(n * p)]).reshape(n , p)
        

    Z_alter = Z

    #print(f"The matrix Z : {Z}")
    #print(f"The mean of matrix Z : {np.mean(Z)}")

    try:
        N_col_Z = Z_alter.shape[1]
    except Exception as error:
        if type(error) == IndexError:
            print(f" You don't have the necessary dimensions, so we will reshape your matrix w !")
            Z_alter = Z_alter.reshape(-1, 1)
            N_col_Z = Z_alter.shape[1]
    
    for i in range(N_col_Z): # iterando sobre cada coluna.
        N_dropped_neurons = int(np.ceil((1 - keep_rate) * Z_alter[:, i].shape[0])) # Pegando o número de neurônios que serão dropados (arredondando pra cima).
        lowest_val_idx = np.array(np.argsort(Z_alter[:, i])[: N_dropped_neurons]) # Pegandos os índices dos neurônios que serão dropados.
        for j in lowest_val_idx: # Zerando os pesos menos relevantes.
            Z_alter[j, i] = 0



    # A saída da rede é obtida com a pseudoinversa.
    H = np.tanh(np.dot(xin, Z_alter))
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)

    w_1 = np.dot(np.transpose(H), H)
    print(f"The first part HTH is {w_1[ :2, :2]}")
    print(f" The mean of HTH is {np.mean(w_1)}")


    w_2 = np.linalg.inv(w_1)
    print(f"The inverse of H-1 is {w_2[ :2, :2]}")
    print(f" The mean of the inverse is {np.mean(w_2)}")
    
    w_3 = np.dot(w_2, np.transpose(H))
    #print(f"The third part H-1 is {w_3[ :5, :5]}")
    
    w = np.dot(w_3, yin)
    #print(f"The last part is {w}")
    
    
    try:
        N_col_W = w.shape[1]
    except Exception as error:
        if type(error) == IndexError:
            print(f" You don't have the necessary dimensions, so we will reshape your matrix w !")
            w = w.reshape(-1, 1)
            N_col_W = w.shape[1]
        


    for i in range(N_col_W): # Removendo os pesos menos relevantes da camada de saída.
        N_dropped_neurons = int(np.ceil((1 - keep_rate) * w[:, i].shape[0])) # Pegando o número de neurônios que serão dropados.
        lowest_val_idx = np.array(np.argsort(w[:, i])[: N_dropped_neurons]) # Pegandos os índices dos neurônios que serão dropados.
        for j in lowest_val_idx: # Zerando os pesos menos relevantes.
            w[j, i] = 0

    # Retornos.
    return_list = list()
    return_list.append(w)   
    return_list.append(H)
    return_list.append(Z) # Conexões são desligadas apenas no treino, portanto tenho que mandar a matriz Z completa.
    return_list.append(Z_alter)
    return  return_list


def test_ELM(xin: np.ndarray, Z: np.ndarray, W: np.ndarray, control: bool):
    
    if control == True:
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin, ones), axis = 1)
        
    H = np.tanh(np.dot(xin, Z))
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)
    
    Y_hat = np.sign(np.dot(H, W)) # Para problemas de classificação.
    #Y_hat = np.dot(H, W) # Para problemas de regressão.
    return Y_hat



# Treinando a rede e obtendo os parâmetros
%%time
p_neurons = 5
ret = train_ELM_DropOut(xin = X_train, yin = y_train, p = p_neurons, keep_rate = 0.8, control = True)
wxor = ret[0]
hxor = ret[1]
zxor = ret[2]
z_alter = ret[3]




# Para plotar a superfície : 
labels_df_reshaped = labels_df_xor.reshape(-1, 1)
mat_plot = np.concatenate((x_df_xor, labels_df_reshaped), axis = 1)
index_sort = 2
sorted_indices = np.argsort(mat_plot[:, index_sort])
mat_plot = mat_plot[sorted_indices]



x1_points = mat_plot[: 150, 0 : 2]
x2_points = mat_plot[150 :, 0 : 2]
plt.scatter(x1_points[:, 0], x1_points[:, 1], color = 'red', label = 'data 1')
plt.scatter(x2_points[:, 0], x2_points[:, 1], color = 'blue', label = 'data 2')
plt.title('data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.plot()



seqx1x2 = np.linspace(start = -4, stop = 4, num = 300)
np_grid = seqx1x2.shape[0]
shape = (np_grid, np_grid)
MZ = np.zeros(shape)
for i in range(np_grid):
    for j in range(np_grid):
        x1 = seqx1x2[i]
        x2 = seqx1x2[j]
        x1x2 = np.column_stack((x1, x2, 1))
        h1 = np.tanh(np.dot(x1x2, z_alter))
        h1 = np.column_stack((h1, np.ones_like(h1[:, 0])))
        MZ[i, j] = np.sign(np.dot(h1, wxor))[0]


plt.contour(seqx1x2, seqx1x2,from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_hatest)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_hatest)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
