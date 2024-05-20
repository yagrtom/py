
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
#Читаем данные из JSON файла

with open(sys.argv[1], 'r') as file:
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
    if(len(sys.argv) == 4):
        if (sys.argv[2] == "Y"):
            step2 = [int(sys.argv[3]), int(sys.argv[4])]
    else:
        step2 = [-2, 2]
    plt.yticks(step2, rotation ='horizontal')

set_yticks_step()
plt.show()
