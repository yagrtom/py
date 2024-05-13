import json
import numpy as np
from scipy.special import spherical_jn, spherical_yn, spherical_in


#Класс для записи (формирования) ответа
class RadarCrossSection:
    def __init__(self, frequency_range, radius):
        self.frequency_range = frequency_range
        self.radius = radius
#Фунция подсчета
    def calculate_rcs(self):
        rcs_data = []
        for freq in self.frequency_range:
            wavelength = 3e8 / freq
            k = 2 * np.pi / wavelength
            a_n = spherical_jn(0, k * self.radius) / spherical_in(0, k * self.radius)
            b_n = (k * self.radius * spherical_jn(0, k * self.radius) - spherical_jn(1, k * self.radius)) / \
                  (k * self.radius * spherical_yn(0, k * self.radius) - spherical_yn(1, k * self.radius))
            rcs = (wavelength*wavelength / np.pi) * np.abs((np.sum([(-1)**n * (n + 0.5) * (b_n - a_n) for n in range(1, 10)]))**2)
            rcs_data.append({"freq": freq, "lambda": wavelength, "rcs": rcs})

        return rcs_data
#класс для вывода результата в файл
class ResultsOutput:
    def __init__(self, data):
        self.data = data

    def save_to_json(self, filename):
        with open(filename, "w") as file:
            json.dump({"data": self.data}, file, indent=4)



#чтение с файла
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
#print (D, fmin, fmax)



if __name__ == "__main__":
    frequency_range = np.linspace(fmin, fmax, 100)  # Frequency range from 0.1 GHz to 10 GHz
    radius = D/2
    rcs_calculator = RadarCrossSection(frequency_range, radius)
    rcs_data = rcs_calculator.calculate_rcs()

    output = ResultsOutput(rcs_data)
    output.save_to_json("results.json")