import json
from scipy.special import jn, yn, hankel2
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
r = D/2

class RCS:
    def __init__(self, freq_range, r):
        self.freq_range = freq_range
        self.r = r

    def calculate_rcs(self):
        lambda_range = [1/fmax, 1/fmin]
        data = []
        for freq in self.freq_range:
            k = 2 * 3.14159 * freq
            rcs = 0
            for n in range(1, 20):
                a_n = jn(n, k*self.r) / hankel2(n, k*self.r)
                b_n = (k*self.r*jn(n-1, k*self.r) - n*jn(n, k*self.r)) / (k*self.r*hankel2(n-1, k*self.r) - n*hankel2(n, k*self.r))
                rcs += (-1)**n * (n + 0.5) * (b_n - a_n)
            rcs = (lambda_range[0]**2 / 3.14159) * abs(rcs)**2
            data.append({"freq": freq, "lambda": lambda_range[0], "rcs": rcs})
        return data

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
        plt.title('RCS vs Frequency')
        plt.grid()
        plt.show()

freq_range = range(1, 11)





rcs_calculator = RCS(freq_range, r)
data = rcs_calculator.calculate_rcs()

output = Output(data)
output.save_to_json('result.json')
output.plot_data()
