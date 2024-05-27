import json
import scipy.constants as constants
import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt


#Чтение из файла
import urllib.request
import xml.etree.ElementTree as ET
url = 'https://jenyay.net/uploads/Student/Modelling/task_rcs.xml'
with urllib.request.urlopen(url) as response:
    dataread = response.read()
tree = ET.fromstring(dataread)
for elem in tree.iter('variant'):
    if elem.attrib['number'] == '10':
        D = float(elem.attrib['D'])
        fmin = float(elem.attrib['fmin'].replace('e9', '')) * 1e9
        fmax = float(elem.attrib['fmax'].replace('e9', '')) * 1e9

freq= range(int(fmin) , int(fmax), 10000000)

class EDA:
    def __init__(self, D, freq):
        self.r = D/2
        self.wave_length = 0
        self.k = 0
        self._freq_range_ = freq
    def a_n(self, n):
        numerator = np.longdouble(special.spherical_jn(n, self.k * self.r))
        divider = self.h_n(n, self.k * self.r)
        return np.divide(numerator, divider)

    def b_n(self, n):
        numerator = self.k * self.r * np.longdouble(special.spherical_jn(n - 1, self.k * self.r)) - n * np.longdouble(special.spherical_jn(n, self.k * self.r))
        divider = self.k * self.r * self.h_n(n - 1, self.k * self.r) - n * self.h_n(n, self.k * self.r)
        return np.divide(numerator, divider) # numpy делитель
    def h_n(self, n, arg):
        return np.clongdouble(special.spherical_jn(n, arg) + 1j*special.spherical_yn(n, arg))
    
    def calculEDA(self):
        coef = self.wave_length**2 / constants.pi
        partForml = 0
        # оператор суммы в формуле c верхним пределом 50
        for n in range(1, 50):
            partForml += (-1)**n * (n + 0.5) * (self.b_n(n) - self.a_n(n))
        result = coef * abs(partForml) ** 2
        return result
    
    def GetData(self):
        self.data = []
        for freq in self._freq_range_:
            # обновляем длину волны и волновое число
            self.wave_length = np.longdouble(constants.speed_of_light / freq)
            self.k = np.longdouble(2 * constants.pi / self.wave_length)
            # получаем значение ЭПР для новых параметров
            temp_eda = self.calculEDA()
            self.data.append({"freq": float(freq), "lambda": float(self.wave_length), "rcs": float(temp_eda)})
        return self.data

class Output:
    def __init__(self, data):
        self.data = data

    def save_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump({"data": self.data}, f, indent=4)

    def plot_data(self):
        freq = [d["freq"] for d in self.data]
        rcs = [d["rcs"] for d in self.data]
        plt.plot(freq, rcs)
        plt.xlabel('Frequency')
        plt.ylabel('RCS')
        plt.title('RCS from frequency')
        plt.grid()
        plt.show()
        
calculator = EDA(D, freq)
data = calculator.GetData()
output = Output(data)
output.save_to_json('result.json')
output.plot_data()
