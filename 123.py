
import numpy as np
import math


# 定义Ackley函数
def f(X):
    dim = len(X)

    t1 = 0
    t2 = 0
    for i in range(0, dim):
        t1 += X[i] ** 2
        t2 += math.cos(2 * math.pi * X[i])

    OF = 20 + math.e - 20 * math.exp((t1 / dim) * -0.2) - math.exp(t2 / dim)

    return OF


from geneticalgorithm import geneticalgorithm as ga

varbound = np.array([[-32.768, 32.768]] * 2)
model = ga(function=f, dimension=2, variable_type='real', variable_boundaries=varbound)
model.run()

# 画图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
ax.plot_surface(X, Y, Z, cmap='jet')

# 写标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X)')
ax.set_title('Ackley Function')
plt.show()
