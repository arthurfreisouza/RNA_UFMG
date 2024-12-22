import numpy as np
import matplotlib.pyplot as plt

# Função geradora que será aproximada por um polinômio de grau 2
def fgx(xin):
    return 1000*np.sin(xin)


# AMOSTRAGEM DOS PONTOS COM 1 ERRO GAUSSIANO INCLUSO.
np.random.seed(123) # Alterando a semente.
X = np.random.uniform(low=-15, high=10, size=20) # Criando as amostras aleatórias de X.
Y = fgx(X) + 4  np.random.randn(len(X)) # Criando as coordenadas Y dos 20 pontos aleatórios do eixo X.
# O resultado do comando acima dará 1 função parecida com a descrita na função fgx, mas conterá 1 erro associado.

# Construção da matriz de design H para um polinômio de grau N
# Cada coluna é uma transformação polinomial de X, e a última coluna é preenchida com uns, representando o termo constante.
H = np.column_stack([X**5, X **4, X**3, X**2, X, np.ones_like(X)])

# A pseudoinversa é usada para encontrar uma solução de mínimos quadrados para sistemas de equações lineares quando a matriz não é invertível.
w = np.linalg.pinv(H) @ Y #  Isso resultará em um vetor w que contém os pesos ou parâmetros do modelo, que melhor ajustam os dados de entrada X aos dados de saída Y de acordo com o modelo especificado.
# w conterá os pesos ajustados, não esquecer que Hw = y


# Criando e plotando os gráficos contínuos para criar 1 representação gráfica.
xgrid = np.linspace(-15, 10, 1000) # Gera 1 distribuição de 1000 pontos que começa em -15 até o 10.
ygrid = fgx(xgrid)
Hgrid = np.column_stack([xgrid**5,xgrid **4, xgrid **3, xgrid**2, xgrid, np.ones_like(xgrid)]) 
yhatgrid = Hgrid @ w # Calculando os valores do y contínuos.

plt.plot(xgrid, ygrid, color='blue', label='Função Original')
plt.scatter(X, Y, color='red', label='Dados Originais')
plt.plot(xgrid, yhatgrid, color='green', label='Função Aproximada')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Aproximação de Função')
plt.legend()
plt.show()

