import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.fft import rfft, rfftfreq, irfft

def mnk(x,y):
    le = len(x)

    k = ((x * y).sum() - x.sum() * y.sum() / le) / ((x * x).sum() - (x.sum())**2 / le)
    b = y.sum() / le - k * x.sum() / le
    #sk = 1 / le ** 0.5 * (((y * y).sum() - (y.sum())**2 / le)/((x * x).sum() - (x.sum())**2 / le) - b**2)**0.5
    sk = 1 / le ** 0.5 * (abs(((y * y).sum() - (y.sum()) ** 2 / le) / ((x * x).sum() - (x.sum()) ** 2 / le) - b ** 2)) ** 0.5
    sb = sk * ((x * x).sum() / le - (x.sum())**2 / le**2) ** 0.5
    return k, b, sk, sb

def sred(x,y, par):
    y = y[:len(y) - len(y) % par]
    return x[:len(x) - par:par], (np.reshape(y, (len(y) // par, par))).sum(axis=1) / par


data = pd.read_csv('Problem_33_file_010.dat', sep='\t', header=None)
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
yf = rfft(y)
spp = x.max() / len(x)
xf = rfftfreq(len(x), spp)
limit = sorted(yf)[len(yf)-2]


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
ax1.scatter(x,y, color = 'b', s=3)
ax2.scatter(xf, yf, color = 'r', s = 5)

freq1 = 0
freq2 = 0
amp1 = 0
amp2 = 0
for i in range(len(yf)):

    if yf[i] < limit:
        yf[i] = 0
    elif freq1 == 0:
        freq1 = xf[i]
        amp1 = yf[i] / len(yf)
    else:
        freq2 = xf[i]
        amp2 = yf[i] / len(yf)

#ax3.scatter(xf, yf, color = 'r', s = 5)
clear = irfft(yf)
#print(len(xf))
ax3.plot(x, clear, color = 'g')
ax4.plot(x[5000:5100], clear[5000:5100], color = 'g')
print('Первый сигнал: ')
print('Амплитуда:', amp1.real)
print('Частота:', freq1, 'Гц')
print()
print('Второй сигнал: ')
print('Амплитуда:', amp2.real)
print('Частота:', freq2, 'Гц')
plt.show()