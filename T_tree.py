import numpy as np

def find_eigen_vectors(matrix: np.ndarray) -> np.ndarray:
    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    mask = np.isclose(eigen_values, 1.0)
    eigen_vectors = eigen_vectors[:, mask]
    return eigen_vectors



matrix = np.random.rand(3**5, 3**5)  # Пример матрицы 3^5 на 3^5
eigen_vectors = find_eigen_vectors(matrix)  # Находим +1 собственные векторы

# Выводим найденные собственные векторы
for i, eigen_vector in enumerate(eigen_vectors.T):
    print(f"Собственный вектор {i + 1}: {eigen_vector}")