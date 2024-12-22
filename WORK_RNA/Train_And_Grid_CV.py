import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def train_ELM(xin : np.ndarray, yin : np.ndarray, p : int, control : bool, lam : float) -> list:
    n = xin.shape[1]
    
    if control == True:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range((n + 1) * p)]).reshape(n + 1, -1)
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin,ones), axis = 1)
    else:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range(n * p)]).reshape(n , -1)

    H = np.tanh(np.dot(xin, Z))
    ones = np.ones((H.shape[0], 1))
    H = np.concatenate((H, ones), axis = 1)

    diagonal_matrix = lam * np.eye(H.shape[1])
    # w = (HT H + λIp)^(-1)HT y
    w1 = np.linalg.inv(np.dot(np.transpose(H), H) + diagonal_matrix)
    w1 = np.linalg.inv(np.dot(np.transpose(H), H) + diagonal_matrix)
    w = np.dot(w1, np.dot(np.transpose(H), yin))

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




def cross_validation(xin : np.ndarray, yin : np.ndarray, p : int, lam : np.ndarray,CV_groups : int) -> float:
    yin = yin.reshape(-1, 1)
    data = np.concatenate((xin, yin), axis = 1)
    acc_arr = np.zeros(CV_groups)
    for i in range(CV_groups):
        np.random.shuffle(data)
        size_group = (data.shape[0] // CV_groups)
        
        train_data = data[ : size_group*(CV_groups - 1), :]
        test_data = data[size_group*(CV_groups - 1): , :]


        ret_ = train_ELM(train_data[:, : train_data.shape[1] - 1], train_data[:, train_data.shape[1] - 1], p, True, lam)
        w = ret_[0]
        H = ret_[1]
        Z = ret_[2]

        y_pred = test_ELM(test_data[:, : test_data.shape[1] - 1], Z, w, True)

        acc_ = accuracy_score(test_data[:, test_data.shape[1] - 1 : ], y_pred)
        acc_arr[i] = acc_
    
    return np.mean(acc_)

def grid_search_lam(xin : np.ndarray, yin : np.ndarray, p : int, lam : np.ndarray, CV_groups : int):

    arr_lam = np.zeros(lam.shape[0])
    arr_acc = np.zeros(lam.shape[0])
    for index, value in enumerate(lam):
        arr_acc[index] = cross_validation(xin = xin, yin = yin, p = p, lam = value, CV_groups = CV_groups)
        arr_lam[index] = value
    idx = np.argmax(arr_acc)
    print(f"The model with best accuracy has the accuracy : {np.max(arr_acc)}")
    print(f"The model parameters with best accuracy has the lambda : {arr_lam[idx]}")
    return arr_lam[idx], np.max(arr_acc)
