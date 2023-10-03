from qiskit import QuantumCircuit
#from qiskit.tools.visualization import circuit_drawer
from qiskit.visualization import *
from qiskit.tools.monitor import *
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

# Создание квантовой схемы
qc = QuantumCircuit(3, 3)

# Применение операций квантовых вентилей
qc.h(range(3))
qc.x(range(3))
qc.z(2)
qc.cx(1, 2)
qc.rz(0.5, range(3))
qc.ry(0.5, range(3))
qc.measure(range(3), range(3))
qc.draw(output='mpl')
# Визуализация схемы
circuit_drawer(qc)