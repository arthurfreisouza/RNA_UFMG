import numpy as np

# Criando 1 matriz de entrada com 4 linhas e 2 colunas.
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
X = X.reshape(4, 2)
#print(X.shape) # Pegando o shape do array de entradas X.

w = np.array([1, 1])
#print(w.shape)

YNN = np.dot(X, w)
#print(YNN)
#print(type(YNN))

def result_OR(arr_):
    for i in range(len(arr_)):
        if int(arr_[i]) > 0.5:
            arr_[i] = 1
        else:
            arr_[i] = 0
    return arr_


def result_AND(arr_):
    for i in range(len(arr_)):
        if int(arr_[i]) > 1.5:
            arr_[i] = 1
        else:
            arr_[i] = 0
    return arr_

yhat_OR = result_OR(YNN.copy())

yhat_AND = result_AND(YNN.copy())

result = np.column_stack((X, yhat_OR, yhat_AND))
names_col = ["in1", "in2", "OR", "AND"]

result = np.vstack((names_col, result))
print(result)