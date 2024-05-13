import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Задаем функцию
def f(x1, x2):
    return (10*2 + x1**2 - 10*np.cos(2*np.pi*x1) + x2**2 - 10*np.cos(2*np.pi*x2))
# Генерируем данные
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f(X1, X2)

# Создаем графики
fig = plt.figure(figsize=(15, 10))

# Поверхность
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X1, X2, Y, cmap='viridis')
ax1.set_title('3D поверхность')

# Поверхность сверху
ax2 = fig.add_subplot(222)
ax2.contourf(X1, X2, Y, levels=20, cmap='viridis')
ax2.set_title('Поверхность "вид сверху"')


#меняем количество точек, для сглаживания графика
x1 = np.linspace(-5, 5, 1000)
x2 = np.linspace(-5, 5, 1000)
# График y = f(x1) при x2 = 0
ax3 = fig.add_subplot(223)
plt.plot(x1, f(x1, 0))
ax3.set_title('График y = f(x1) при x2 = 0')

# График y = f(x2) при x1 = 0
ax4 = fig.add_subplot(224)
plt.plot(x2, f(0, x2))
ax4.set_title('График y = f(x2) при x1 = 0')

plt.tight_layout()
plt.show()
