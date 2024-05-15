import json
import numpy as np
import matplotlib.pyplot as plt
import os

A = 1.25313
n = 10000
x = [0]*n
fx = [0]*n
data = []
dx = 0.001
a = -100
b = 100
print("Entering values")
for i in range(a, b+1):
    x[i] = i
    fx[i] = 0.5 + (((np.cos(np.sin((x[i]**2) - (A**2))))**2) - 0.5)/( 1 + 0.001*((x[i]**2) + (A**2))) 
    print("x = ",x[i],"             fx = ",fx[i])
    data.append({"x": x[i], "y": fx[i]})
print("The value is entered")


print("Write it to a file")
result = {"data":data}
if not os.path.isdir("result"):
     os.mkdir("result")
with open('result/result.json', "w") as file:
    json.dump(result, file, indent=4)
print("File test.json has been created successfully!")


print("Drawing a graph")
fig = plt.subplot()
ax = plt.subplot()
ax.set_title('Graf')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim((-5, 5))
ax.grid()
x = np.linspace(-100, 100, n)
y = 0.5 + (((np.cos(np.sin((x**2) - (A**2))))**2) - 0.5)/( 1 + 0.001*((x**2) + (A**2)))
ax.plot(x, y)
plt.show()



