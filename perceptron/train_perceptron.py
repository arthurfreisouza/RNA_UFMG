
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def train_perceptron(x_inp : np.array, yi : np.array, learn_rate : float, tol : float, max_epochs : int, control_var : bool): # Fazendo o treinamento do perceptron.
    dim = list(x_inp.shape)
    try : # Caso eu tenha algum problema com as colunas do meu programa...
        n_rows = dim[0]
        n_cols = dim[1]
    except Exception as error:
        if error == "IndexError":
            print("Now, you don't have cols, so we will change it...\n")
            n_cols = 1
        else:
            print(f"The error {error} is hapenning \n")
            print("Breaking the program...")
            sys.exit()
    finally:
        if control_var == True: # control_var é 1 variável de controla que controlará quando usarei um certo threshold...
            w = (np.random.uniform(size = n_cols + 1) - 0.5) # Inicializando o pesos com o tamanho n_cols + 1.
            ones = np.ones((n_rows, 1))
            x_inp = np.concatenate((x_inp, ones), axis = 1) # Apenas colocando as colunas no vetor de entrada.
        else:
            w = (np.random.uniform(size = n_cols) - 0.5)
        n_epochs = 0
        err_epoch = tol + 1
        lst_errors = np.zeros((max_epochs))
        lst_outs = np.zeros((n_rows))
        aux = 0
        while ((n_epochs < max_epochs) and (err_epoch > tol)):
            error_grad = 0
            rand_order = np.random.permutation(n_rows)
            for i in range(n_rows):
                # Escolhendo uma entrada aleatória.
                i_rand = rand_order[i]
                x_val = x_inp[i_rand, :]
                y_hat = 1 if np.dot(x_val, w) >= 0 else 0 # A saída separadora do perceptron.
                err = (yi[i_rand] - y_hat)
                dw = (learn_rate*err*x_inp[i_rand, :])
                w = w + dw # Atualização de pesos.
                if n_epochs == max_epochs - 1:
                    lst_outs[aux] = y_hat
                    aux += 1
                error_grad = error_grad + (err**2)
            lst_errors[n_epochs] = error_grad / n_rows 
            n_epochs += 1
    return (w, lst_errors, lst_outs)


def yperceptron(x_input : np.array, w : np.array, control_var : bool):
    try : 
        n_rows = x_input.shape[0]
        n_cols = x_input.shape[1]
    except Exception as error:
        print(f"The error {error} is happening ...")
        n_cols = 1
        x_input = x_input.reshape(-1, 1)
    if control_var == True:
        ones = np.ones((n_rows, 1))
        x_inp = np.concatenate((x_inp, ones), axis = 1) # Apenas colocando as colunas no vetor de entrada.
    u = np.dot(x_input, w)
    
    y = np.where(u >= 0, 1, -1) # Compara elemento a elemento com 0, retorna 1 caso maior e 0 caso menor.
    return y    
