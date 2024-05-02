import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

data = pd.read_csv('Problem_03_file_025.dat', sep='\t', header=None)
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
ax1.scatter(x, y, color='b', s=5)

xu, xd = x.max(), x.min()
l = len(x)
#xm, ym = sred(x, y, l // 100)
#ax2.scatter(xm, ym, color='b', s=5)
coef = mnk(x, y)
k = coef[0]
b = coef[1]
sk = coef[2]
sb = coef[3]
yt = k * x + b
ax1.scatter(x, yt, color='r', s=5)
print('сновная зависимость')
print('k =',k, '+-',sk, '|', 'eps_k =', sk / abs(k) * 100, '%')
print('b =',b, '+-',sb, '|', 'eps_b =', sb / abs(b) * 100, '%')
#ax1.title('P1 = 0.99, depolarize')
#plt.legend()
#ax2.scatter(x, y - yt, color='b', s=5)
noise1 = ax2.hist(y - yt, bins=40)
#print(y - yt)
def if_normal(noise):
    n_, amp_ = noise[0], noise[1][:-1]
    n = []
    amp = []
    for i in range(len(n_)):
        if n_[i] != 0:
            n.append(n_[i])
            amp.append(amp_[i])
    amp = np.array(amp)
    n = np.array(n)
    n = n / n.sum()
    #print(len(n))
    #print(len(n))
    kn,bn, skn, sbn = mnk(amp ** 2, np.log(n))
    sig = (((amp**2 - kn * amp **2 - bn) **2).sum()) ** 0.5 / len(amp) / (-(amp ** 2).min() + (amp ** 2).max())
    if sig < 0.5:
        ax3.scatter(amp ** 2, np.log(n), color='b', s=7)
        lx = np.linspace((amp ** 2).min(), (amp ** 2).max(), 1000)
        ly = kn * lx + bn
        ax3.scatter(lx, ly, color='r', s=1)
        print('Гаусов шум')
        print('mu = 0')
        sg = (-2 * kn) ** -0.5

        ly = -1/2/sg**2 * lx -np.log(sg * (2 * np.pi)**2)
        ax3.scatter(lx, ly, color='g', s=1)

        print('sigma = ', sg, 'chek', abs((bn +np.log(sg * (2 * np.pi)**2)) / bn * 100), '%')
        #sg = (-kn / 2) ** -0.5
        ax4.scatter(amp, n, color='b', s=7)
        #print(n.sum(), (1/ (sg * (2 * np.pi)**0.5) * np.exp(-1 * amp ** 2 / ( 2 * sg**2)) * (-amp_.min() + amp_.max()) / len(amp)).sum())
        ax4.scatter(amp, 1/ (sg * (2 * np.pi)**0.5) * np.exp(-1 * amp ** 2 / ( 2 * sg**2)) * (-amp_.min() + amp_.max()) / len(amp), color = 'r', s = 3)
if_normal(noise1)

plt.show()
