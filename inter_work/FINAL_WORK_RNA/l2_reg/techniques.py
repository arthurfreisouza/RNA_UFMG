import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import sys


def train_ELM_L2_REG(xin : np.ndarray, yin : np.ndarray, p : int, control : bool, lam : float) -> list:
    try:
        if yin.shape[1] == 1:
            pass
    except IndexError:
        yin = yin.reshape(-1, 1)
    try:
        if xin.shape[1] == 1:
            pass
    except IndexError:
        xin = xin.reshape(-1, 1)

    n = xin.shape[1]
 
    if control == True:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range((n + 1) * p)]).reshape(n + 1, -1)
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin,ones), axis = 1)
    else:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range(n * p)]).reshape(n , -1)

    # Z[n ou n + 1, p]
    '''if control == True:
        Z = np.zeros((n+1, p))
        for i in range(p):
            np.random.seed(np.random.randint(0, 10000))
            random_num = np.random.rand()
            np.random.seed(np.random.randint(0, 10000))
            col = np.random.random(n + 1)
            Z[: , i ] = random_num*col
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin,ones), axis = 1)
    else:
        for i in range(p):
            np.random.seed(np.random.randint(0, 10000))
            random_num = np.random.rand()
            np.random.seed(np.random.randint(0, 10000))
            col = np.random.random(n)
            Z[: , i ] = random_num*col'''


    # Fazendo a matriz H, que será 1 função sigmoidal sobre as entradas e os pesos intermediários aleatórios.
    H = np.tanh(np.dot(xin, Z))
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)

    # Realizando a regularização e adicionando o termo de regularização, alterando a função de custo : w = ((HT)H + λIp)^(-1)(HT)y.
    diagonal_matrix = lam * np.eye(H.shape[1])
    w1 = np.linalg.inv(np.dot(np.transpose(H), H) + diagonal_matrix)
    w = np.dot(w1, np.dot(np.transpose(H), yin))

    return_list = []
    return_list.append(w)   
    return_list.append(H)
    return_list.append(Z)
    return  return_list


def test_ELM(xin: np.ndarray, Z: np.ndarray, W: np.ndarray, control: bool, classification : bool):
    
    try:
        if xin.shape[1] == 1:
            pass
    except IndexError:
        xin = xin.reshape(-1, 1)

    if control == True:
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin, ones), axis = 1)

    H = np.tanh(np.dot(xin, Z))
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)

    if classification == True:
        Y_hat = np.sign(np.dot(H, W)) # Para problemas de classificação.
    else:
        Y_hat = np.dot(H, W) # Para problemas de regressão.

    return Y_hat



############################################ grid search and cross_validation for L2 regression ###############################################################



def grid_searchCV_L2(xin : np.ndarray, yin : np.ndarray, p : int, lam : np.ndarray, CV_groups : int, classification : bool):
    arr_lam = np.zeros(lam.shape[0])

    if classification == True:
        arr_acc = np.zeros(lam.shape[0])
        for index, value in enumerate(lam):
            arr_acc[index] = cross_validation_L2(xin = xin, yin = yin, p = p, lam = value, CV_groups = CV_groups, classification = True)
            arr_lam[index] = value
        idx = np.argmax(arr_acc)
        print(f"The model with best accuracy has the mean accuracy : {arr_acc[idx]}")
        print(f"The model parameters with best accuracy is using lambda : {arr_lam[idx]}")
        return arr_lam[idx], np.max(arr_acc), arr_acc
    else:
        arr_MSE = np.zeros(lam.shape[0])
        for index, value in enumerate(lam):
            arr_MSE[index] = cross_validation_L2(xin = xin, yin = yin, p = p, lam = value, CV_groups = CV_groups, classification = False)
            arr_lam[index] = value
        idx = np.argmin(arr_MSE)
        print(f"The model with lowest MSE is : {arr_MSE[idx]}")
        print(f"The model parameters with lowest MSE is using lambda : {arr_lam[idx]}")
        return arr_lam[idx], np.min(arr_MSE), arr_MSE


def cross_validation_L2(xin : np.ndarray, yin : np.ndarray, p : int, lam : float, CV_groups : int, classification : bool) -> float:

    # Caso particular, transforma de (N,) para (N, 1) e dará para ser concatenado sem resultar em erros.
    try:
        if yin.shape[1] == 1:
            pass
    except IndexError:
        yin = yin.reshape(-1, 1)
    try:
        if xin.shape[1] == 1:
            pass
    except IndexError:
        xin = xin.reshape(-1, 1)


    # Concatenando a matriz de entrada x com a saída y para realizar o cross-validation.
    data = np.concatenate((xin, yin), axis = 1)
    acc_arr = np.zeros(CV_groups)
    sum_out = np.zeros(CV_groups)
    np.random.shuffle(data) # Irá randomizar a sequência de dados para deixar o processo mais estocástico
    size_group = (data.shape[0] // CV_groups) # Tamanho de cada grupo.
    
    # Realizando o Cross-Validation.
    for i in range(CV_groups):

        if i == (CV_groups - 1): # último valor.
            train_data = data[ : i*size_group, :]
            test_data = data[i*size_group : , :]
        elif i == 0:
            train_data = data[size_group:, :]
            test_data = data[ :size_group, : ]
        else:
            train_data = np.concatenate((data[ : i*size_group, : ], data[(i + 1)*size_group : , :]), axis = 0)
            test_data = data[i*size_group : (i + 1)*size_group, : ]

        # Treinando e testando.
        ret_ = train_ELM_L2_REG(train_data[:, : train_data.shape[1] - 1], train_data[:, train_data.shape[1] - 1], p, True, lam)
        W_CV = ret_[0]
        H_CV = ret_[1]
        Z_CV = ret_[2]

        if classification == True:
            y_pred = test_ELM(xin = test_data[:, : test_data.shape[1] - 1], Z = Z_CV, W = W_CV, control = True, classification = True)

            try:
                if y_pred.shape[1] == 1:
                    pass
            except IndexError:
                y_pred = y_pred.reshape(-1, 1)

            acc_ = accuracy_score(test_data[:, test_data.shape[1] - 1 : ], y_pred)
            acc_arr[i] = acc_

        else:
            y_pred = test_ELM(xin = test_data[:, : test_data.shape[1] - 1], Z = Z_CV, W = W_CV, control = True, classification = False)

            try:
                if y_pred.shape[1] == 1:
                    pass
            except IndexError:
                y_pred = y_pred.reshape(-1, 1)

            sum_out[i] = np.sum((test_data[:, test_data.shape[1] - 1 :] - y_pred)**2)/y_pred.shape[0]
            #print(f"test data : {test_data[:, test_data.shape[1] - 1 :]}\n - ydata : \n{y_pred}, \nlam : {lam}")
    if classification == True:
        return np.mean(acc_arr) # Retornará a acurácia média.
    else:
        return np.mean(sum_out)
