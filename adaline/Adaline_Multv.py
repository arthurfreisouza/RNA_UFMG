# Implementando o modelo de neurônio adaline e o seu treino.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
# Definição do modelo de neurônio Adaline e os seus respectivos parâmetros.
def Train_Adaline(x_inputs : np.ndarray, real_out : np.ndarray, learning_hate : float, tol :float, max_epoch : int, control_var : bool) -> list:
    
    # Pegando as dimensões da matriz de entrada, as linhas são as N entradas do meu modelo, enquanto as colunas são n dados de cada entrada.
    dimentions = list(x_inputs.shape)
    try:
        N_inputs = dimentions[0]
        n_val_inputs = dimentions[1]
    except Exception as error:
        print(f"The error : {error} is happening. \n") 
        print("Probably, it's happening becausa n_val_inputs = 0, so we change it to 1.")
        n_val_inputs = 1
        print(f"Now, n_val_inputs = {n_val_inputs}")

    # Control_var é uma variável de controle para caso seja necessário adicionar 1 threshold.
    if control_var == True:
        # wt é uma matriz com n_val_inputs + 1 linhas representando valores de parametros inicalizados aleatoriamente.
        # Isso acontece porque vamos acrescentar 1 coluna de 1's na matriz de pesos.
        wt = (np.random.uniform(size = n_val_inputs + 1) - 0.5)
        aux = np.column_stack([np.ones_like(x_inputs[: , 1])])
        x_inputs = np.column_stack([x_inputs, aux])
    else:
        # Caso o sinal de controle seja False, então não terá o threshold e a matriz de pesos continuará intacta.
        wt = (np.random.uniform(size = n_val_inputs) - 0.5)
    

    n_epochs = 0 # Cada iteração de treino sobre o conjunto de dados.
    erro_epoch = tol + 1 # Variável para controlar o loop
    lst_param = np.zeros((max_epoch, len(wt)))
    lst_error_grad = np.zeros((max_epoch))

    while((n_epochs < max_epoch) and (erro_epoch > tol)):
        
        error_grad = 0
        changing_order = np.random.permutation(N_inputs)
        # Irei realizar esse loop enquanto houver dados de entrada, os valores de cada entrada estão aleatórios, para melhorar a performance de treino do modelo.
        for i in range(N_inputs):
            i_aleatorio = changing_order[i] # Escolhi 1 entrada aleatória.
            x_val_al = x_inputs[i_aleatorio,] # Pegando os valores de 1 entrada aleatória.
            wt = wt.reshape(-1, 1)

            y_hat_out = x_val_al @ wt # ŷ = t(X)w, é o transposto de X pois tanto x quanto w estão inicializados como vetores.
            err = real_out[i_aleatorio,] - y_hat_out

            # Utilizando a Regra Delta para a atualização de pesos.
            dw = (learning_hate * err * x_inputs[i_aleatorio, ])
            dw = np.transpose(dw)
            wt = np.add(wt, dw) # w(t + 1) = w(t) + nex
            error_grad = error_grad + (err * err) # Acumulando os erros do gradiente descendente em 1 variável.

        # Para cada epoch, estou adicionando na lista tanto parametros, quanto o erro medio do gradiente.
        #lst_param[n_epochs] = wt
        lst_error_grad[n_epochs] = error_grad / N_inputs
        n_epochs += 1
    # Retornarei 2 listas, sendo cada uma delas contendo cada erro e a outra contendo os parametros.
    arr_return = [lst_error_grad, wt]
    return arr_return




t = np.arange(start = 0, stop = 2*np.pi, step = 0.1*np.pi)

x1 = np.matrix([np.sin(t) + np.cos(t)])
x1 = x1.reshape((-1, 1))
x2 = np.matrix([np.tanh(t)])
x2 = x2.reshape((-1, 1))
x3 = np.matrix([np.sin(4*t)])
x3 = x3.reshape((-1, 1))
x4 = np.matrix([np.abs(np.sin(t))])
x4 = x4.reshape((-1, 1))


y = x1 + 2*x2 + 0.8*x3 + 3.2*x4 + np.pi/2
x = np.column_stack([x1, x2, x3, x4])


# Treinando o meu modelo, irei ter 1 retorno 1 vetor de erros e os parâmetros do meu modelo.
grad_err, param = Train_Adaline(x, y, 0.01, 0.01, 50, True)




# Criando dados de teste e testando o modelo.
t_test = np.arange(start = 0, stop = 2*np.pi, step = 0.01*np.pi)
x1_test = np.matrix([np.sin(t_test) + np.cos(t_test)])
x1_test = x1.reshape((-1, 1))
x2_test = np.matrix([np.tanh(t_test)])
x2_test = x2.reshape((-1, 1))
x3_test = np.matrix([np.sin(4*t_test)])
x3_test = x3.reshape((-1, 1))
x4_test = np.matrix([np.abs(np.sin(t_test))])
x4_test = x4.reshape((-1, 1))
x_test = np.column_stack([x1_test, x2_test, x3_test, x4_test, np.ones_like(x1_test)])
y_test = x_test @ param
print(f"{x_test.shape} * {param.shape} = {y_test.shape}")


fig, axs = plt.subplots(1)
continuous_time = np.linspace(start=0, stop=2*np.pi, num=1000)

x1_plot = np.matrix([np.sin(continuous_time) + np.cos(continuous_time)])
x1_plot = x1_plot.reshape((-1, 1))  # Ensure correct reshaping
x2_plot = np.matrix([np.tanh(continuous_time)])
x2_plot = x2_plot.reshape((-1, 1))  # Ensure correct reshaping
x3_plot = np.matrix([np.sin(4*continuous_time)])
x3_plot = x3_plot.reshape((-1, 1))  # Ensure correct reshaping
x4_plot = np.matrix([np.abs(np.sin(continuous_time))])
x4_plot = x4_plot.reshape((-1, 1))  # Ensure correct reshaping
y_plot = x1_plot + 2*x2_plot + 0.8*x3_plot + 3.2*x4_plot + np.pi/2

plt.plot(continuous_time, y_plot, color='red')
plt.title("Approximation senoidal_part")
plt.xlabel("t")
plt.ylabel("y")

# Ensure that y_test is reshaped to match the shape of t_test
arange_time = np.arange(start = 0, stop = 2*np.pi, step = 0.1*np.pi)
y_test = np.transpose(y_test.ravel())
y_test = np.array(y_test)
y_test = y_test.reshape(-1, 1)
plt.scatter(arange_time, y_test, color='blue', s = 10)
plt.plot(arange_time, y_test, color='blue', linestyle='-')
plt.title("Approximation senoidal_part")
plt.xlabel("t")
plt.ylabel("y")
plt.show()