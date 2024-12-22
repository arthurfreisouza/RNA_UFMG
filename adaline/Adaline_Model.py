import numpy as np
import matplotlib.pyplot as plt
import time


def train_adaline(xin :np.array, yd : np.array, eta : float, tol : float, maxepochs : int, par : int): # xin é 1 vetor de entrada que conterá todas as entradas da rede neural.
    dimentions = list(xin.shape) # Pegando as dimensões da minha rede neural, N conterá o número de linhas e o n terá o numero de colunas
    try : 
        N = dimentions[0] # Numero de linhas da rede neural, ou seja, número de entrada de dados que gerará 1 saída yhat.
        n = dimentions[1] # Número de colunas da rede neural, cada coluna representará 1 neurônio da camada de entrada.
    except:
        print("There is not a n value here...")
        n = 1
        print("Now n is 1!")

        

    if par == 1: # par é um sinal que indicará quando terá ou não o threshold.
        wt = np.random.uniform(size = n + 1) - 0.5 # Inicializando os pesos aleatóriamente e subtraindo 0.5 para dar mais aleatoriedade. (n + 1) pois há o threshold.
        xin = np.column_stack([np.ones_like(xin), xin]) # Adicionando 1 nova coluna fixa em 1's para a matrix de entrada.
    else:
        wt = np.random.uniform(size = n) - 0.5
        #wt = np.array([np.random.uniform(size = n) - 0.5]) # Inicializando os pesos aleatóriamente, agora nao há a camada de threshold.
        
    nepochs = 0 # Contador que irá indicar o número de epocas, irá até um valor limite máximo.
    erro_epoch = tol + 1
    evec = np.zeros((maxepochs)) # Array que conterá cada erro do gradiente para cada entrada.
    dist_param = np.zeros((maxepochs))
    
    print(wt)
    print(wt.shape)
    time.sleep(10)
    print(xin[1, : ])
    print(xin[1, : ].shape)
    time.sleep(10)
    while(nepochs < maxepochs) and (erro_epoch > tol): # Loops para parar o while.
        error_grad = 0 # Inicializando a variável que conterá os erros do gradiente descendente para cada epoch.
        x_seq = np.random.permutation(N) # Irei embaralhar os valores de 0 até o N de entradas.
        for i in range(N):
            irand = x_seq[i] # Pegando 1 valor aleatório da sequência aleatória que indicará um X.
            x_vec = xin[irand, : ] # O vetor de entrada será a linha aleatória definida pela permutação.
            yhat = np.dot(np.transpose(x_vec), wt) # Calculando a saída da rede (usando a regra de Hebbs) para 1 linha aleatória definida pelo permutation.
            ei = (yd[irand] - yhat) # Calculando o erro da rede.
            dw = (eta * ei * xin[irand, : ]) # Pegando o termo (n.e.x) da equação da regra delta.
            wt = wt + dw # Usando a regra delta w(t + 1) = w(t) + n.ei.xi
            aux = np.sqrt(wt[0]**2 + wt[1]**2)
            error_grad = error_grad + (ei * ei) # Calculando o erro do gradiente descendente.

        dist_param[nepochs] = aux
        evec[nepochs] = error_grad / N # Adicionando os erros médios em 1 vetor que conterá os erros.
        nepochs += 1 # Incrementando o número de épocas.

    print(f"The final parameters are {wt}")
    retlist = np.array((dist_param, evec)) # Criando 1 lista com os pesos e os erros médios de cada peso.
    return (retlist, wt)


#(x(t), y(t))
t = np.arange(start = 0, stop = 0.2*np.pi, step = 0.1*np.pi)
x = np.array(np.sin(t))
y = np.array(4*x + 2) # Função aproximadoras

learning_hate = float(input(" Write the learning hate, a good value is 0.015 ..."))
n_epochs =  int(input(" Write the total epochs, put values greather than 1000 to see a good result ..."))
retlist, param = train_adaline(x, y, learning_hate, 0.01, n_epochs, 1)
print(param)




x2 = np.linspace(start = 0, stop = 2*np.pi, num = len(retlist[1]))
#x2 = np.linspace(start = 0, stop = n_epochs, num = 1000)
y2 = (4 * x2 + 2)
y_hat = (param[1]*x2 + param[0])




fig, axs = plt.subplots(2, 2)


axs[0,0].plot(x2, y2, color = 'red')
axs[0,0].plot(x2, y_hat, color = 'blue')
axs[0,0].legend(['function_generator => y = 4*x + 2', f'function_aproximator => y = {round(param[1], 4)}*x + {round(param[0])}'])
axs[0,0].set_title("Aproximation")
axs[0,0].set_xlabel("x")
axs[0,0].set_ylabel("y")



axs[0,1].scatter(x2, retlist[1], color = 'black', s = 5)
axs[0,1].set_xlabel("Epoch")
axs[0,1].set_ylabel("Error")
axs[0,1].set_title("Error decay")



# Criando os dados de teste x.

t_test = np.arange(start = 0, stop = 2*np.pi, step = 0.01*np.pi)
x_test = np.sin(t_test)
x_test = np.column_stack([np.ones_like(x_test), x_test]) # Lembrar que a matriz de parametros é uma matriz de linhas, contendo 2 linhas. Portanto, Ncol1 = Nrow2
y_model = (x_test @ param)


axs[1,0].scatter(t_test, y_model, color = 'red', s = 5)
axs[1,0].plot(t_test, y_model, color = 'red')
axs[1,0].plot(t_test, 4 * np.sin(t_test) + 2, color = 'black')
axs[1,0].set_title("Graph")
axs[1,0].set_xlabel("t_test")
axs[1,0].set_ylabel("aproximation sin(t_test)")

plt.show()