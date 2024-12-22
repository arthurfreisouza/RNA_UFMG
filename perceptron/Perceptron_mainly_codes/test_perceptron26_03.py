# Implementando o modelo de neurônio adaline e o seu treino.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
# Definição do modelo de neurônio Adaline e os seus respectivos parâmetros.
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
            x_inp = np.column_stack([x_inp, np.ones_like(x_inp[:, 0])]) # Apenas colocando as colunas no vetor de entrada.
        else:
            w = (np.random.uniform(size = n_cols) - 0.5)
        n_epochs = 0
        err_epoch = tol + 1
        lst_errors = np.zeros((max_epochs))
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
                error_grad = error_grad + (err**2)
            lst_errors[n_epochs] = error_grad / n_rows 
            n_epochs += 1
    return (w, lst_errors)





xc1 = (0.3*np.random.normal(size = 60) + 2).reshape(-1, 2)
xc2 = (0.3*np.random.normal(size = 60) + 4).reshape(-1, 2)


xall = np.vstack((xc1, xc2))
zeros = np.zeros(xc1.shape[0])
ones = np.ones(xc1.shape[0])
yall = np.concatenate((zeros, ones), axis=0)
control_var = bool(input("Write the True or False to the control var."))
retlist = []
retlist = train_perceptron(xall, yall, 0.01, 0.2, 10000, control_var)
w = retlist[0]
lst_errors = retlist[1]

t = np.arange(start = 0, stop = 10, step = 0.001)
x_ = np.linspace(start = 0, stop = 10, num = 1000)
y_ = -(w[0] * x_ + w[2]) / w[1]

epochs = range(1, len(lst_errors) + 1)

plt.scatter(xc1[:, 0], xc1[:, 1], color = 'blue', label = "Sample1")
plt.scatter(xc2[:, 0], xc2[:, 1], color = 'red', label = "Sample2")
plt.plot(x_, y_, color = 'orange', label = 'straight')
plt.legend()
plt.show()


