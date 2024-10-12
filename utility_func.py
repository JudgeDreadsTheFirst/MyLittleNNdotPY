import numpy as np

def clip_by_norm(grad, max_norm = 1.0):
    grad_norm = np.linalg.norm(grad)
    if grad_norm > max_norm:
        grad = max_norm * grad / grad_norm
    return grad

def high_correlation_filter(data, corr_matrix,corr_threshold = 1):
    """
    Функция реализации фильтра высокой корреляции.
    :param data: Матрица признаков
    :param corr_matrix: Матрица коэффициентов корреляции
    :param corr_threshold: Пороговое значение фильтрации
    :return: Отфильтрованная матрица
    """
    indices = np.where(np.abs(corr_matrix)> corr_threshold)
    feature_remove = set()

    # отбираем признаки пропуская индекса i==j (ну типа логично, данные сами с собой скоррелированы)
    for i, j in zip(*indices):
        if i != j and (j,i) not in feature_remove:
            feature_remove.add((i, j))

    feature_remove = list(feature_remove)

    data_filtered = np.delete(data, [j for i,j in feature_remove], axis = 1)

    print('Размерность данных после фильтрации:', data_filtered.shape)
    return data_filtered