import numpy as np
import matplotlib.pyplot as plt



# Criando a variável t, que será um intervalo contínuo.
t = np.linspace(start = 0.2*np.pi, stop = 0.5*np.pi, num = 100) 
# A variável x seŕa uma função senoidal que será aplicada sobre o intervalo contínuo de tempo. Irei guardar todos os valores do intervalo em 1 array.
x = np.array(np.sin(t))
# Irei aplicar, sobre a entrada senoidal, uma função y = -2x.
y = np.array(-2*x)
# Acrescentando  uma coluna de 1's no meu array x de entrada (lembrar que terei o threshold).
x = np.column_stack([x, np.ones_like(t)])
print(x)
# Extraindo os termos da diagonal desse produto matricial. Isso é para normalizar os dados.
dK = np.diag(x @ np.transpose(x)) # pegando os elementos da diagonal da matriz de autocorrelação a = (WX^T.X).
x[:, 0] = x[:, 0] / dK
x[:, 1] = x[:, 1] / dK
w = np.transpose(x) @ y

print(w)