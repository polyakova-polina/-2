import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


data = pd.read_csv('Problem_14_file_052.dat', sep='\t', header=None)
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
xm, ym = sred(x,y,15)
mera = 0
for i in range(len(y) - 1):
    mera += (y[i+1] - y[i])**2
mera = mera ** 0.5 / (len(y) - 1)
deg = 1
while deg < 100:
    coef = np.polyfit(x, y, deg)
    #print(coef)
    yt = np.zeros((1, len(y)))[0]
    for i in range(deg + 1):
        yt += coef[deg - i] * x**i
    srkv = (((y - yt)**2).sum())**0.5 / len(y)
    print(srkv)
    if srkv / mera < 1:
        break
    deg += 1

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
ax1.scatter(x, y, color='b', s=5)
ax1.scatter(x, yt + srkv, color='g', s=5)
ax1.scatter(x, yt - srkv * np.ones_like(yt), color='g', s=5)

xu, xd = x.max(), x.min()
#xm, ym = sred(x,y,15)
#ax1.scatter(x, yt, color='r', s=5)
x_ = []
y_ = []
yt_ = []
#for i in range(len(x)):

'''
sns.distplot(y - yt, hist=True, kde=False,
bins=int(180/5), color = 'blue',
hist_kws={'edgecolor':'black'})
'''
#print(y - yt)
noise1 = ax2.hist(y-yt,bins = 50, color = 'blue', edgecolor = 'black')

def if_normal(noise):
    n_, amp_ = noise[0], noise[1][:-1]
    n = []
    amp = []
    for i in range(len(n_)):
        if n_[i] >2:
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

        ly = -1/2/sg**2 * lx -np.log(sg * (2 * np.pi)**2) / 2
        ax3.scatter(lx, ly, color='g', s=1)

        print('sigma = ', sg, 'chek', abs((bn +np.log(sg * (2 * np.pi)**2)/2) / bn * 100), '%')
        #sg = (-kn / 2) ** -0.5
        ax4.scatter(amp, n, color='b', s=7)
        #print(n.sum(), (1/ (sg * (2 * np.pi)**0.5) * np.exp(-1 * amp ** 2 / ( 2 * sg**2)) * (-amp_.min() + amp_.max()) / len(amp)).sum())
        ax4.scatter(amp, 1/ (sg * (2 * np.pi)**0.5) * np.exp(-1 * amp ** 2 / ( 2 * sg**2)) * (-amp.min() + amp.max()) / len(amp), color = 'r', s = 3)
if_normal(noise1)
#print(len(noise1[1]))
#ax3.scatter(noise1[0], noise1[1])
#print(y - yt)
plt.show()
