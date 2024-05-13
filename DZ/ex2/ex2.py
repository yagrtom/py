import json
import numpy as np
import matplotlib.pyplot as plt

A = 1.25313
n = 5000
x = [0]*n
fx = [0]*n
data = []
a = -100
b = 100
#выводим значения X и соответствующие ему значения F(x)
print("Entering values")
for i in range(0, n):
    x[i] = a - i*((a-b)/n)
    fx[i] = 0.5 + (((np.cos(np.sin((x[i]**2) - (A**2))))**2) - 0.5)/( 1 + 0.001*((x[i]**2) + (A**2))) 
    print("x = ",x[i],"             fx = ",fx[i])
    data.append({"x": x[i], "y": fx[i]})
print("The value is entered")

#записываем данные в файл
print("Write it to a file")
result = {"data":data}
with open("result.json", "w") as file:
    json.dump(result, file, indent=4)
print("File test.json has been created successfully!")

#рисуем график по функции
print("Drawing a graph")
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






#Читаем данные из JSON файла
inputname = input("Write name file >")
with open(inputname, 'r') as file:
    data = json.load(file)

x_values = []
y_values = []

#Извлекаем данные из JSON
for point in data['data']:
    x_values.append(point['x'])
    y_values.append(point['y'])

#Строим график
graf_from_json = plt.subplot()
plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('y')
graf_from_json.set_xlim((-5, 5))
plt.title('Graph of y = f(x)')
def set_yticks_step():
    step = [int(i) for i in input("Write diapozone Y >").split()]
    plt.yticks(step, step, rotation ='horizontal')
set_yticks_step()
plt.show()