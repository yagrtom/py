import matplotlib.pyplot as plt
import numpy as np
import math

A = 1.25313

fig = plt.subplot()
ax = plt.subplot()
ax.set_title('Graf')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim((-5, 5))
ax.grid()
x = np.linspace(-100, 100, 10000)
y = 0.5 + (((np.cos(np.sin((x**2) - (A**2))))**2) - 0.5)/( 1 + 0.001*((x**2) + (A**2)))
ax.plot(x, y)
plt.show()