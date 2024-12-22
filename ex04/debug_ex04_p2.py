import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

def train_adaline(x_inputs : np.array, yd : np.array, learning_rate : float, tol : float, maxepochs : int, control_var : bool):
    dimentions = list(x_inputs.shape)
    try:
        N_inputs = dimentions[0]
        n_val_inputs = dimentions[1]
    except Exception as error:
        print(f" The error {error} is happening. It's happening because n_val_inputs has 0 columns.")
        print(f" So, we will change it to 1")
        print(f"Changing ...")
        n_val_inputs = 1
        print(f"Now, n_val_inputs is {n_val_inputs}")
    finally:
        if control_var == True:
            
            w = np.random.uniform(size = n_val_inputs + 1) - 0.5
            # Estou organizando minha entrada x em colunas, e adicionando 1 coluna extra de 1s.
            aux = np.column_stack([np.ones_like(x_inputs[:, 1])])
            x_inputs = np.column_stack([x_inputs, aux])
        else:
            w = np.random.uniform(size = n_val_inputs) - 0.5

        n_epochs = 0 # É o número de vezes que estou treinando usando TODOS os dados de entrada.
        erro_epoch = tol + 1
        lst_errors_grad = np.zeros((maxepochs))

        # Loop while que resultará no treino do meu modelo.
        while ((n_epochs < maxepochs) and (erro_epoch > tol)):
            erro_grad = 0

            # Alterando a ordem de dados de treino, no fito de o gradiente descendente não ficar estático em 1 lugar específico.
            change_order_train = np.random.permutation(N_inputs)
            for i in range(N_inputs):
                i_rand = change_order_train[i]
                x_val_train = x_inputs[i_rand, : ]
                # Não é necessário fazer o np.transpose(), pois já fiz implicitamente através do column_stacks.
                y_hat = np.dot(x_val_train, w) #ŷ = ([X] @ w)
                err = (yd[i_rand] - y_hat)
                dw = (learning_rate*err* x_inputs[i_rand, :])
                w = w + dw
                erro_grad = erro_grad + (err * err)
                
            lst_errors_grad[n_epochs] = erro_grad / N_inputs
            n_epochs += 1
        return (w, lst_errors_grad)
    
t = np.array(pd.read_csv('t', delimiter = ' '))
x = pd.read_csv('x', delimiter = ' ')
y = np.array(pd.read_csv('y', delimiter = ' '))
x1 = np.array(x['V1'])
x2 = np.array(x['V2'])
x3 = np.array(x['V3'])
x = np.column_stack([x1, x2, x3])
y_plot = x1 + x2 + x3

w, lst_err_grad = train_adaline(x, y, learning_rate = 0.01, tol = 0.01, maxepochs = 20, control_var = True)