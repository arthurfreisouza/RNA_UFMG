import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys



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

    return_list = []
    return_list.append(w)   
    return_list.append(H)
    return_list.append(Z)
    return  return_list







def train_ELM_PRUNING(xin : np.ndarray, yin : np.ndarray, p : int, keep_rate : float, control : bool) -> list:
    np.random.seed(np.random.randint(0, 10000))

    n = xin.shape[1] # Pegando o número de valores de cada entrada.

    # Z[n ou n + 1, p]
    if control == True:
        Z = np.zeros((n+1, p))
        #Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range((n + 1) * p)]).reshape(n + 1, p)
        for i in range(p):
            np.random.seed(np.random.randint(0, 10000))
            random_num = np.random.rand()
            np.random.seed(np.random.randint(0, 10000))
            col = 100*np.random.random(n + 1)
            Z[: , i ] = random_num*col
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin,ones), axis = 1)
    else:
        for i in range(p):
            #Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range((n) * p)]).reshape(n, p)
            np.random.seed(np.random.randint(0, 10000))
            random_num = np.random.rand()
            np.random.seed(np.random.randint(0, 10000))
            col = 100*np.random.random(n)
            Z[: , i ] = random_num*col
            

    
    H = np.tanh(np.dot(xin, Z))
    scaler = StandardScaler()
    H = scaler.fit_transform(H)

    
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)
    #print(f"det : {np.linalg.det(np.dot(np.transpose(H), H))}")

    w = np.dot(np.linalg.pinv(H), yin) # w = H⁺y
    
    try:
        N_col_W = w.shape[1]
    except IndexError:
        w = w.reshape(-1, 1)
        N_col_W = w.shape[1]


    for i in range(N_col_W): # Removendo os pesos menos relevantes da camada de saída.
        N_dropped_neurons = int(np.ceil((1 - keep_rate) * w[:, i].shape[0])) # Pegando o número de neurônios que serão dropados.
        idx = np.array(np.argsort(np.abs(w[:, i]))) # Pegandos os índices dos neurônios que serão dropados.
        lowest_val = idx[:N_dropped_neurons]
        for j in lowest_val: # Zerando os pesos menos relevantes.
            w[j, i] = 0


    return_list = list()
    return_list.append(w)   
    return_list.append(H)
    return_list.append(Z) # Conexões são desligadas apenas no treino, portanto tenho que mandar a matriz Z completa.
    return  return_list


def test_ELM(xin: np.ndarray, Z: np.ndarray, W: np.ndarray, control: bool):
    
    if control == True:
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin, ones), axis = 1)
    H = np.tanh(np.dot(xin, Z))
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)
    Y_hat = np.sign(np.dot(H, W))
    #Y_hat = np.dot(H, W) # Para problemas de regressão.

    return Y_hat



############################################ grid search and cross_validation for L2 regression ###############################################################



def grid_searchCV_L2(xin : np.ndarray, yin : np.ndarray, p : int, lam : np.ndarray, CV_groups : int):
    arr_lam = np.zeros(lam.shape[0])
    arr_acc = np.zeros(lam.shape[0])
    for index, value in enumerate(lam):
        arr_acc[index] = cross_validation_L2(xin = xin, yin = yin, p = p, lam = value, CV_groups = CV_groups)
        arr_lam[index] = value
        #print(f"The model accuracy : {arr_acc[index]}, using lam : {arr_lam[index]}\n")
    idx = np.argmax(arr_acc)
    print(f"The model with best accuracy has the mean accuracy : {arr_acc[idx]}")
    print(f"The model parameters with best accuracy is using lambda : {arr_lam[idx]}")
    return arr_lam[idx], np.max(arr_acc)



def cross_validation_L2(xin : np.ndarray, yin : np.ndarray, p : int, lam : float, CV_groups : int) -> float:

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


###############################################################################################################################################################






############################################## grid search and cross validation for pruning #####################################################################


def grid_searchCV_pruning(xin : np.ndarray, yin : np.ndarray, p : int, keep_rate : np.ndarray, CV_groups : int):
    arr_krate = np.zeros(keep_rate.shape[0])
    arr_acc = np.zeros(keep_rate.shape[0])
    for index, value in enumerate(keep_rate):
        arr_acc[index] = cross_validation_pruning(xin = xin, yin = yin, p = p, k_rate = value, CV_groups = CV_groups)
        arr_krate[index] = value
        #print(value)
        #print(f"The model accuracy : {arr_acc[index]}, using lam : {arr_lam[index]}\n")
    idx = np.argmax(arr_acc)
    print(f"The model with best accuracy has the mean accuracy : {arr_acc[idx]}")
    print(f"The model parameters with best accuracy is using keep_rate : {arr_krate[idx]}")
    return arr_krate[idx], np.max(arr_acc)


def cross_validation_pruning(xin : np.ndarray, yin : np.ndarray, p : int, k_rate : float, CV_groups : int) -> float:

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
        #ret_ = train_ELM_PRUNING_new_method(xin = train_data[:, : train_data.shape[1] - 1], yin = train_data[:, train_data.shape[1] - 1], p = p, control = True, keep_rate = k_rate)
        ret_ = train_ELM_PRUNING(xin = train_data[:, : train_data.shape[1] - 1], yin = train_data[:, train_data.shape[1] - 1], p = p, control = True, keep_rate = k_rate)

        W_CV = ret_[0]
        H_CV = ret_[1]
        Z_CV = ret_[2]
        #print(test_data[:, : test_data.shape[1] - 1])
        y_pred = test_ELM(test_data[:, : test_data.shape[1] - 1], Z_CV, W_CV, True)
        #print(f"ypred = {y_pred}")
        #print(f"z : {Z_CV}")
        #print(f"w : {W_CV}")

        # Montando um array de acurácias.
        acc_ = accuracy_score(test_data[:, test_data.shape[1] - 1 : ], y_pred)
        acc_arr[i] = acc_
    
    return np.mean(acc_arr) # Retornará a acurácia média.


###############################################################################################################################################################


































def train_ELM_PRUNING_new_method(xin : np.ndarray, yin : np.ndarray, p : int, keep_rate : float, control : bool) -> list:
    np.random.seed(np.random.randint(0, 10000))

    n = xin.shape[1] # Pegando o número de valores de cada entrada.

    # Z[n ou n + 1, p]
    if control == True:
        Z = np.zeros((n+1, p))
        for i in range(p):
            np.random.seed(np.random.randint(0, 10000))
            random_num = np.random.rand()
            np.random.seed(np.random.randint(0, 10000))
            col = 10*np.random.random(n + 1)
            Z[: , i ] = random_num*col
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin,ones), axis = 1)
    else:
        for i in range(p):
            np.random.seed(np.random.randint(0, 10000))
            random_num = np.random.rand()
            np.random.seed(np.random.randint(0, 10000))
            col = 10*np.random.random(n)
            Z[: , i ] = random_num*col
            

    
    H = np.tanh(np.dot(xin, Z))
    scaler = StandardScaler()
    H = scaler.fit_transform(H)

    
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)



    ####################### LOGICA PRA W ############################################
    tol = 0.05
    num_epoch = 0
    max_epoch = 100
    err_epoch = tol + 1 
    eta = 0.01
    N = xin.shape[0]
    n_pruning = 10
    size_epoch_prun = max_epoch / n_pruning
    ################################# INICIALIZANDO OS PESOS DE W ALEATORIAMENTE #################
    try:
        if yin.shape[1] == 1:
            pass
    except IndexError:
        yin = yin.reshape(-1, 1)

    w = np.random.random(size = (p + 1)*yin.shape[1]).reshape(p + 1, yin.shape[1]) - 0.5
    N_col_W = w.shape[1]

    while((num_epoch < max_epoch) and (err_epoch > tol)):
        sequence = np.arange(start = 0, stop = N, step = 1).tolist()
        seq_randomized = random.sample(population = sequence, k = N)
        ei2 = 0

        for i in range(N): # Calculando a saída para todas as entradas.
            rand_input = seq_randomized[i]
            inpt_outp_layer = H[rand_input, :]
            try:
                if inpt_outp_layer.shape[1] == 1:
                    pass
            except IndexError:
                inpt_outp_layer = inpt_outp_layer.reshape(-1, 1)

            inpt_outp_layer_T = np.transpose(inpt_outp_layer)
            y_net = np.sign(np.dot(inpt_outp_layer_T, w))

            err = (yin[i, :] - y_net)
            #print(f"outp wished : {yin[i, :]}, real output {y_net}, err: {err}")
            w = w + eta*err*inpt_outp_layer # Atualização dos pesos

            if (num_epoch % size_epoch_prun) == 0:
                for j in range(N_col_W): # Removendo os pesos menos relevantes da camada de saída.
                    N_dropped_neurons = int(np.ceil((1 - keep_rate) * w[:, j].shape[0])) # Pegando o número de neurônios que serão dropados.
                    idx = np.array(np.argsort(np.abs(w[:, j]))) # Pegandos os índices dos neurônios que serão dropados.
                    lowest_val = idx[:N_dropped_neurons]
                    for k in lowest_val: # Zerando os pesos menos relevantes.
                        w[k, j] = 0
            ei2 = ei2 + (err*err)
        err_epoch = (ei2 / N)

        num_epoch = num_epoch + 1                
    return_list = list()
    return_list.append(w)   
    return_list.append(H)
    return_list.append(Z) # Conexões são desligadas apenas no treino, portanto tenho que mandar a matriz Z completa.
    return  return_list





