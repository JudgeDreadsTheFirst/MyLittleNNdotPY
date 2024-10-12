import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from NN_classes import *
import utility_func

#TODO: разбить все на дополнительные подфункции
def Linear_Regression_with_Grad_Descent():

    xtr = np.load('./X_train.npy')
    ytr = np.load('./y_train.npy')
    print(f"Размеры массива значений целевой переменной:{ytr.shape}")
    print(f"Размеры массива признаков объектов:{xtr.shape}")

    colors = ['red', 'blue', 'yellow', 'green']

    plt.figure() # создаем фигуру
    for i in range(1, xtr.shape[1]+1): # в цикле i будет являтся порядковым номером слайса данных
        plt.subplot(2, 2 , (i, i)) # разбиваем фигуру на четыре области и в каждой по порядку отрисовываем признак
        plt.scatter(xtr[:, i - 1], ytr, s = 1, c = colors[i - 1])
        plt.title(f"slice {i - 1}", fontdict = {'fontsize' : 10}) # задаем название графику в активной области
    plt.tight_layout() # убираем наложение осей и названий графиков
    plt.savefig(f"feature slices.svg")  # сохраняем полученную фигуру в картинку
    plt.show()

    correlation = np.corrcoef(xtr, rowvar=False)
    sns.heatmap(correlation, annot = True, fmt = ".5f", cmap = "coolwarm") # визуализируем матрицу коэффициентов
    plt.title("Тепловая карта корреляционных коэффициентов")
    plt.savefig(f"heatmap_prefilt.svg") # сохраняем картинку
    plt.show()
    print(f"Корреляция данных в массиве признаков\n {correlation}")

    #xtr = utility_func.high_correlation_filter(xtr, correlation, corr_threshold=1) #использовалось для проверки прогона без фильтрации
    xtr = utility_func.high_correlation_filter(xtr, correlation, corr_threshold=0.5)

    correlation = np.corrcoef(xtr, rowvar=False)
    sns.heatmap(correlation, annot=True, fmt=".5f", cmap="coolwarm")
    plt.title("Тепловая карта корреляционных коэффициентов")
    plt.savefig(f"heatmap_postfilt.svg")
    plt.show()

    network = NN()
    mu = network(xtr)
    loss_fn = Loss()
    loss_value = loss_fn(mu, ytr)
    USG = loss_fn.backward()
    grad = network.backward(usg=USG)

    learning_rate = 1e-4
    epochs = 10000

    loss_history = []
    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        # как все пиздато сделано, осталость совсем чуть чуть
        # сделать блять все классы
        #TODO: повеситься
        #TODO: ну по факту "реализовать" надо только Linear и чуть чуть MSE
        mu = network(xtr)
        loss_value = loss_fn(mu, ytr)
        USG = loss_fn.backward()
        grad = network.backward(usg=USG)
        grad = utility_func.clip_by_norm(grad, 10)

        # update network parameters
        network.l1.theta = network.l1.theta - grad*learning_rate
        loss_history.append(loss_value)
        pbar.update(1)
        pbar.set_postfix({'loss': loss_value})

    pbar.close()
    # отобразите эволюцию функции потерь по мере обучения сети
    plt.plot(loss_history)
    plt.yscale('log')

    # примените нейросеть к данным Xtr
    mu = network(xtr)
    # отобразите диаграмму y(y_true) для оценки соответствия полученного решения известному
    plt.scatter(ytr, mu, s=1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Linear_Regression_with_Grad_Descent()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
