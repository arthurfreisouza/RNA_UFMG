import numpy as np
import matplotlib.pyplot as plt
import sys



def train_ELM(xin : np.ndarray, yin : np.ndarray, p : int, control : bool) -> list:
    n = xin.shape[1]
    #print(xin.shape)
    #sys.exit()
    if control == True:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range((n + 1) * p)]).reshape(n + 1, -1)
        ones = np.ones((xin.shape[0], 1))
        xin = np.concatenate((xin,ones), axis = 1)
    else:
        Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range(n * p)]).reshape(n , -1)

    H = np.tanh(np.dot(xin, Z))
    ones = np.ones((H.shape[0], 1))

    H = np.concatenate((H, ones), axis = 1)

    w = np.dot(np.linalg.pinv(H), yin)
    
    
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