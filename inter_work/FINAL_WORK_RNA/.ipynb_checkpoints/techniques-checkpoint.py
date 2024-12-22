import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def train_ELM_L2_REG(xin : np.ndarray, yin : np.ndarray, p : int, control : bool, lam : float) -> list:
    n = xin.shape[1]
    
    if control == True:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range((n + 1) * p)]).reshape(n + 1, -1)
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin,ones), axis = 1)
    else:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range(n * p)]).reshape(n , -1)

    # Fazendo a matriz H, que será 1 função sigmoidal sobre as entradas e os pesos intermediários aleatórios.
    H = np.tanh(np.dot(xin, Z))
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)

    # Realizando a regularização e adicionando o termo de regularização, alterando a função de custo : w = ((HT)H + λIp)^(-1)(HT)y.
    diagonal_matrix = lam * np.eye(H.shape[1])
    w1 = np.linalg.inv(np.dot(np.transpose(H), H) + diagonal_matrix)
    w = np.dot(w1, np.dot(np.transpose(H), yin))
    #print(f"matrix w {w}")

    return_list = []
    return_list.append(w)   
    return_list.append(H)
    return_list.append(Z)
    return  return_list



def test_ELM(xin: np.ndarray, Z: np.ndarray, W: np.ndarray, control: bool):
    
    if control == True:
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin, ones), axis = 1)
    H = np.tanh(np.dot(xin, Z))
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)
    Y_hat = np.sign(np.dot(H, W))
    return Y_hat



def grid_searchCV(xin : np.ndarray, yin : np.ndarray, p : int, lam : np.ndarray, CV_groups : int):
    arr_lam = np.zeros(lam.shape[0])
    arr_acc = np.zeros(lam.shape[0])
    for index, value in enumerate(lam):
        arr_acc[index] = cross_validation(xin = xin, yin = yin, p = p, lam = value, CV_groups = CV_groups)
        arr_lam[index] = value
        #print(f"The model accuracy : {arr_acc[index]}, using lam : {arr_lam[index]}\n")
    idx = np.argmax(arr_acc)
    print(f"The model with best accuracy has the mean accuracy : {arr_acc[idx]}")
    print(f"The model parameters with best accuracy is using lambda : {arr_lam[idx]}")
    return arr_lam[idx], np.max(arr_acc)



def cross_validation(xin : np.ndarray, yin : np.ndarray, p : int, lam : float, CV_groups : int) -> float:

    # Caso particular, transforma de (N,) para (N, 1) e dará para ser concatenado sem resultar em erros.
    try:
        if yin.shape[1] == 1:
            pass
    except IndexError:
        yin = yin.reshape(-1, 1)


    # Concatenando a matriz de entrada x com a saída y para realizar o cross-validation.
    data = np.concatenate((xin, yin), axis = 1)
    acc_arr = np.zeros(CV_groups)
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
        y_pred = test_ELM(test_data[:, : test_data.shape[1] - 1], Z_CV, W_CV, True)

        # Montando um array de acurácias.
        acc_ = accuracy_score(test_data[:, test_data.shape[1] - 1 : ], y_pred)
        acc_arr[i] = acc_
    
    return np.mean(acc_arr) # Retornará a acurácia média.