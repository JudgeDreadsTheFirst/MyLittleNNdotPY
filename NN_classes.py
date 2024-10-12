
#INFO: и вот как будто бы все ясно, а вроде нихуя не ясно
#TODO: добавить комментарии и описания классов
#TODO: накидать assert из уважения к offensive programming
class Differentiable:
    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def backward(self, **kwargs):
        raise NotImplementedError()


class Residual(Differentiable):
    """
    D(residual) = mu - y
    mu - vector DIM: Nx1
    y - vector DIM: Nx1
    """
    def __init__(self):
        super(Residual, self).__init__()
        self.cache = None

    def __call__(self, mu, y):
        return self.forward(mu, y)

    def forward(self, mu, y):
        # Этот метод реализует вычисление отклонения mu-y
        d = mu - y
        self.cache = [1, 1] # [NxN identity, NxN identity]

        return d

    def backward(self, usg):
        # Этот метод реализует вычисление градиента отклонения D по аргументу mu
        assert self.cache is not None, "Backward pass before forward pass!!!"

        partial_grad = self.cache
        self.cache = None

        return partial_grad


class MSE(Differentiable):
    def __init__(self):
        super(MSE, self).__init__()
        self.cache = None

    def __call__(self, d):
        return self.forward(d)

    def forward(self, d):
        # Этот метод реализует вычисление значения функции потерь
        # Подсказка: метод должен возвращать единственный скаляр - значение функции потерь
        self.cache = None
        mse_value = None

        return mse_value

    def backward(self):
        # Этот метод реализует вычисление градиента функции потерь по аргументу d
        # Подсказка: метод должен возвращать вектор градиента функции потерь
        #           размерностью, совпадающей с размерностью аргумента d
        assert self.cache is not None, "Backward pass before forward pass!!!"

        partial_grad = None

        ### YOUR CODE HERE
        # partial_grad = ...

        return partial_grad


class linear(Differentiable):
    def __init__(self):
        super(linear, self).__init__()
        self.theta = None
        self.cache = None

    def __call__(self, X):
        # этот метод предназначен для вычисления значения целевой переменной
        return self.forward(X)

    def forward(self, X):
        # этот метод предназначен для применения модели к данным
        assert X.ndim == 2, "X should be 2-dimensional: (N of objects, n of features)"

        # ВНИМАНИЕ! Матрица объекты-признаки X не включает смещение
        #           Вектор единиц для применения смещения нужно присоединить самостоятельно!

        ### YOUR CODE HERE
        # X_ = ...

        if (self.theta is None):
            # Если вектор параметров еще не инициализирован, его следует инициализировать
            # Подсказка: длина вектора параметров может быть получена из размера матрицы X
            # Fx1.T dot NxF.T = 1xN
            # Если X - матрица объекты-признаки, то это матрица из вектор-строк!
            self.theta = None

        # Здесь следует собственно применить модель к входным данным

        z = None
        self.cache = None

        ### YOUR CODE HERE
        # z = ...
        # self.cache = ...

        return z

    def backward(self, usg):
        # Этот метод реализует вычисление компоненты градиента функции потерь

        assert self.cache is not None, "please perform forward pass first"

        partial_grad = None
        self.cache = None

        ### YOUR CODE HERE
        # partial_grad = ...

        # Не забудьте очистить кэш!
        # self.cache = ...

        return partial_grad


class Identity(Differentiable):
    def __init__(self):
        super(Identity, self).__init__()

    def __call__(self, X):
        # этот метод предназначен для вычисления значения функции активации
        return self.forward(X)

    def backward(self, usg):
        # Этот метод реализует вычисление компоненты градиента функции потерь
        return usg

    def forward(self, X):
        # этот метод предназначен для вычисления функции активации
        return X


class NN(Differentiable):
    def __init__(self):
        super(NN, self).__init__()
        self.l1 = linear()
        self.act = Identity()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        mu = None
        # Этот метод будет вычислять нейросеть на данных X
        ### YOUR CODE HERE
        mu = self.act(self.l1(X))
        return mu

    def backward(self, usg):
        grad = None
        ### YOUR CODE HERE
        usg_act = self.act.backward(usg)
        grad = self.l1.backward(usg_act)
        return grad


class Loss(Differentiable):
    def __init__(self):
        super(Loss, self).__init__()
        self.dev = Residual()
        self.mse = MSE()

    def __call__(self, mu, y):
        return self.forward(mu, y)

    def forward(self, mu, y):
        l = None
        # Этот метод будет вычислять нейросеть на данных X
        ### YOUR CODE HERE
        d = self.dev(mu, y)
        l = self.mse(d)
        # TODO: выход должен быть скаляр, добавить assert
        return l

    def backward(self, usg):
        grad = None
        ### YOUR CODE HERE
        usg_mse = self.mse.backward(usg)
        grad = self.dev.backward(usg_mse)
        return grad