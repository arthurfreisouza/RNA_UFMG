import numpy as np
import matplotlib.pyplot as plt

N = 30

xc1 = (0.6 * np.random.normal(size = N * 2)).reshape(-1, 2)
xc1[: , 0] = xc1[: , 0] + 2
xc1[: , 1] = xc1[: , 1] + 2

xc2 = (0.6 * np.random.normal(size = N * 2)).reshape(-1, 2)
xc2[: , 0] = xc2[: , 0] + 4
xc2[: , 1] = xc2[: , 1] + 4


xc3 = (0.6 * np.random.normal(size = N * 2)).reshape(-1, 2)
xc3[: , 0] = xc3[: , 0] + 2
xc3[: , 1] = xc3[: , 1] + 4

xc4 = (0.6 * np.random.normal(size = N * 2)).reshape(-1, 2)
xc4[: , 0] = xc4[: , 0] + 4
xc4[: , 1] = xc4[: , 1] + 2

g1 = np.concatenate((xc1, xc2), axis = 0)
g2 = np.concatenate((xc3, xc4), axis = 0)


plt.scatter(g1[:, 0], g1[:, 1], color = 'red', label = 'data1')
plt.scatter(g2[:, 0], g2[:, 1], color = 'blue', label = 'data2')
plt.legend()
plt.show()

X = np.concatenate((g1, g2), axis = 0)
ones_vec = np.ones_like(g1[:, 0])
Y = np.concatenate((ones_vec, -ones_vec), axis = 0)


p = 100
Z = np.array([np.random.uniform(-0.5, 0.5) for _ in range(3 * p)]).reshape(3, -1)
Xaug = np.column_stack((X, np.ones_like(X[:, 0])))
H = np.tanh(np.dot(Xaug, Z))


Haug = np.column_stack((H, np.ones_like(H[:, 0])))
w = np.dot(np.linalg.pinv(Haug), Y)

Y_hat_train = np.sign(np.dot(Haug, w))
errors = Y - Y_hat_train
print(np.sum(errors))


seqx1x2 = np.linspace(start = -2, stop = 10, num = 300)
np_grid = seqx1x2.shape[0]
shape = (np_grid, np_grid)
MZ = np.zeros(shape)
for i in range(np_grid):
    for j in range(np_grid):
        x1 = seqx1x2[i]
        x2 = seqx1x2[j]
        x1x2 = np.column_stack((x1, x2, 1))
        h1 = np.tanh(np.dot(x1x2, Z))
        h1 = np.column_stack((h1, np.ones_like(h1[:, 0])))
        MZ[i, j] = np.sign(np.dot(h1, w))


plt.contour(seqx1x2, seqx1x2, MZ, levels = 1)
plt.scatter(g1[:, 0], g1[:, 1], color = 'red', label = 'data1')
plt.scatter(g2[:, 0], g2[:, 1], color = 'blue', label = 'data2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour Plot')
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.grid(True)
plt.show()