from qiskit import QuantumCircuit, execute, Aer
import math


def grover_algorithm(f, num_iterations):
    # Создание квантовой схемы с n кубитами и n классическими битами
    n = int(math.log2(len(f)))
    circuit = QuantumCircuit(n, n)

    # Применение операции Адамара ко всем кубитам
    circuit.h(range(n))

    # Применение операции оракула f
    oracle = create_oracle(f)
    circuit += oracle

    # Применение операции Диффьюзии
    diffusion = create_diffusion()

    # Применение операции Диффьюзии num_iterations раз
    for _ in range(num_iterations):
        circuit += diffusion

    # Измерение всех кубитов
    circuit.measure(range(n), range(n))

    # Запуск схемы на симуляторе
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1)
    result = job.result()

    return result.get_counts()


# Функция для создания оракула f
def create_oracle(f):
    n = int(math.log2(len(f)))
    circuit = QuantumCircuit(n)

    # Применение операции X к целевому кубиту, если f(x) = 1
    for i in range(len(f)):
        if f[i] == '1':
            circuit.x(i)

    # Применение операции Z к целевому кубиту
    circuit.z(range(n))

    # Применение операции X к целевому кубиту, если f(x) = 1
    for i in range(len(f)):
        if f[i] == '1':
            circuit.x(i)

    return circuit


# Функция для создания операции Диффьюсии
def create_diffusion():
    n = int(math.log2(len(f)))
    circuit = QuantumCircuit(n)

    # Применение операции Адамара ко всем кубитам
    circuit.h(range(n))

    # Применение операции X ко всем кубитам
    circuit.x(range(n))

    # Применение операции Z к контрольному кубиту
    circuit.h(n - 1)
    circuit.mct(list(range(n - 1)), n - 1)
    circuit.h(n - 1)

    # Применение операции X ко всем кубитам
    circuit.x(range(n))

    # Применение операции Адамара ко всем кубитам
    circuit.h(range(n))

    return circuit


# Пример использования алгоритма для функции f(x) = '1010' и 1 итерации
f = '1010101010101010'
num_iterations = 4
counts = grover_algorithm(f, num_iterations)
print(counts)