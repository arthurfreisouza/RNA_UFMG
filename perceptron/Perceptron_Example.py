# Implementando o modelo de neurônio adaline e o seu treino.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
# Definição do modelo de neurônio Adaline e os seus respectivos parâmetros.
def train_perceptron(x_inputs : np.array, yd : np.ndarray, learning_rate : float, tol : float, max_epochs : int, control_var : bool):
    dim = list(x_inputs.shape)
    try : 
        n_rows = dim[0] # Número de dados de entrada.
        n_cols = dim[1] # Número de valores dos dados de entrada.
    except Exception as error:
        print(f"The error {error} is happening.")
        if error == "IndexError":
            n_cols = 1
        else:
            sys.exit()
    finally : 
        if control_var == True: # Indicará que terá 1 threshold relacionado.
            w = np.random.uniform(size = n_cols + 1) - 0.5 # Criando o vetor de peso aleatoriamente, ele conterá 1 coluna a mais.
            x_inputs = np.column_stack([x_inputs, np.ones_like(x_inputs[: , 0])])
        else:
            w = np.random.uniform(size = n_cols) - 0.5 # Criando o vetor de pesos aleatóriamente e não conterá a coluna de pesos a mais.

        n_epochs = 0
        err_epoch = tol + 1
        lst_errors = np.zeros((max_epochs))

        while((n_epochs < max_epochs) and (err_epoch > tol)):
            err_grad = 0
            order_changed = np.random.permutation(n_rows) # PERMUTAR LINHAS E NAO COLUNAS
            for i in range(n_cols):
                i_rand = order_changed[i]
                x_val = x_inputs[i_rand, : ]
                y_hat = 1 if np.dot(x_val, w) >= 0 else 0
                err = (yd[i_rand] - y_hat)
                dw = (learning_rate*err*x_inputs[i_rand, :])
                w = w + dw
                err_grad = err_grad + (err ** 2)
            lst_errors[n_epochs] = err_grad / n_cols
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