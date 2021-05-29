import math
from datetime import time

import numpy as np
from scipy import sparse as sparse
import matplotlib.pyplot as plt
import numpy.linalg as alg

import loadMNIST
import loadMNIST_V2


def regularLSforQ2a(A, b, lam, G):
    return np.asarray(alg.inv(A.T @ A + lam * G.T @ G) @ A.T @ b)


def Q2a(b, G):
    n = len(b)
    A = np.eye(n)
    lam = 80 / 2
    return regularLSforQ2a(A, b, lam, G)


def IRLS(f, x, y, G, n, lam, epsilon):
    size = len(x)
    # convert to matrix
    A = np.eye(size)
    b = y
    G = sparseToArray(G)
    W = np.eye(size - 1)

    for k in range(0, n - 1):
        # solve x
        x = alg.inv(A.T @ A + lam * G.T @ W @ G) @ (A.T @ b)

        # set W:
        # init w vector as a 0 vector
        w_vec = np.zeros(size)
        # for each row/column (W is diagonal)  - set w_vector as ( 1 / (abs((G @ x)[j]) + epsilon))
        for j in range(0, size - 1):
            w_vec[j] = 1 / (abs((G @ x)[j]) + epsilon)
            W[j][j] = w_vec[j]

    return x


def sparseToArray(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix


def fForVector(x):
    f = np.zeros(x.shape)
    f[0:one] = 0.0 + 0.5 * x[0:one]
    f[one:2 * one] = 0.8 - 0.2 * np.log(x[100:200])
    f[(2 * one):3 * one] = 0.7 - 0.3 * x[(2 * one):3 * one]
    f[(3 * one):4 * one] = 0.3
    f[(4 * one):(5 * one)] = 0.5 - 0.1 * x[(4 * one):(5 * one)]
    return f


def coronaDataToVector(path):
    vectorTuple = []
    with open(path, encoding="utf-8") as coronaData:
        row = coronaData.readline()
        while row != "":
            row = row.removesuffix('\n')
            vectorTuple.append(int(row))
            row = coronaData.readline()
    vectorArray = np.asarray(vectorTuple)
    return vectorArray


def sigmoid(XT, w):
    matrix_mul = -XT @ w
    return 1.0 / (1.0 + np.exp(matrix_mul))


def LRobjective(w, X, C):
    m = len(C[0])
    c1 = np.asarray([C[0]])
    # print(f'c1:\n{c1} \n')

    c2 = np.asarray([C[1]])

    print(f'sigmoid:{(sigmoid(X.T, w))} \n')
    oneVec = np.asarray([[1], [1], [1], [1], [1]])
    print(f' 1- sig = {1.0 - (sigmoid(X.T, w))}')
    print(f' 1- sig with vec = {oneVec - (sigmoid(X.T, w))}')
    print(f'log:\n{np.log(1.0 - sigmoid(X.T, w))}  ')

    # print(f'c1.T:\n{c1} ')
    print(f'!!!!!!:\n{c2 @ np.log(1 - sigmoid(X.T, w))} \n ')

    res = (-1 / m) * ((c1 @ np.log(sigmoid(X.T, w))) + (c2 @ np.log(1 - sigmoid(X.T, w))))
    print(f'result:\n{res}\n')
    return res[0]


def LRGradient(w, X, C):
    m = len(C[0])
    c1 = np.asarray([C[0]])
    print(f' grad, c = {c1.T}')
    res = (1 / m) * X @ (sigmoid(X.T, w) - c1.T)
    print(f'res = {res}')
    return res


def unitVectorGenerator(vecSize):
    v = np.asarray([np.random.randn(vecSize)])
    normalized_V = v / np.linalg.norm(v)
    return normalized_V.T


def LRHessian(X, w, m):
    elementWise = np.multiply(sigmoid(X.T, w), (1 - sigmoid(X.T, w)))
    D = np.zeros((len(elementWise), len(elementWise)))
    for i in range(len(elementWise)):
        D[i][i] = elementWise[i]
    return (1 / m) * X @ D @ X.T


def gradTest(f, X, gradX, sampleVec):
    d = sampleVec
    epsilons = []
    for i in range(1, 90):
        # epsilons.append(0.0015*i)
        epsilons.append(0.1 * i)
    epsilons = [1]
    # print(epsilons)
    Oone = lambda epsilon: abs(f(X + epsilon * d) - f(X))
    print(gradX)
    Otwo = lambda epsilon: abs(f(X + epsilon * d) - f(X) - epsilon * d.T @ gradX)[0][0]
    ones = []
    twos = []

    for i in range(len(epsilons)):
        ones.append(Oone(epsilons[i]))
        twos.append(Otwo(epsilons[i]))

    plt.figure()
    plt.title("g=|f(x+e*d) - f(x)|")
    plt.xlabel("epsilons")
    plt.ylabel("g(e)")
    plt.plot(epsilons[:], ones)

    plt.figure()
    plt.title("g=|f(x+e*d) - f(x)- e * d.T @ grad(x)|")
    plt.xlabel("epsilons")
    plt.ylabel("g(e)")
    plt.plot(epsilons[:], twos)
    # plt.legend(["ones", "twos"])
    plt.show()


def hessianTest(gradFunc, hesFunc, x, sampleVec):
    print("")


def load(n_pictures):
    # valid input
    if n_pictures > 60000 | n_pictures == 0:
        print(f'number of picture is {n_pictures}>60000 or 0')
        return

    # loading data from MNIST
    loader = loadMNIST_V2.MnistDataloader("./dataset/MNIST/train-images.idx3-ubyte",
                                          "./dataset/MNIST/train-labels.idx1-ubyte",
                                          "./dataset/MNIST/t10k-images.idx3-ubyte",
                                          "./dataset/MNIST/t10k-labels.idx1-ubyte")
    imageData = np.asarray(loader.load_data()[0][0])
    imageLabels = np.asarray(loader.load_data()[0][1])

    # flatten the vector
    print(f'before reshaping, imageData shape is: {imageData.shape}')
    imageData = imageData.reshape(-1, 784)
    print(f'after reshaping, imageData shape is: {imageData.shape}')

    # normalizing the vector
    imageData = imageData / 255

    # announcements
    print(f'starting algorithm algorithm with {n_pictures} pictures...')
    # startTime = time.time()

    # kmeans algorithm
    return imageData[0:n_pictures], imageLabels[0:n_pictures]


if __name__ == '__main__':
    Q2 = False
    Q3 = False
    Q4 = True

    if (Q2):
        print(f'-------Q2:------')
        # init - given
        x = np.arange(0, 5, 0.01)
        n = np.size(x)
        one = int(n / 5)
        f = fForVector(x)
        G = sparse.spdiags([[-np.ones(n - 1)], [np.ones(n)]], [-1, 0], n, n)
        G = sparse.spdiags([-np.ones(n), np.ones(n)], np.array([0, 1]), n - 1, n).toarray()
        etta = 0.1 * np.random.randn(np.size(x))
        y = f + etta
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, f)
        plt.show()

        # our part
        xLS = Q2a(y, G)
        xIRLS = IRLS(f, x, y, G, 10, 1, 0.001)
        plt.figure()
        plt.plot(xLS)
        plt.plot(xIRLS)
        plt.show()
        print(f'-----Q2 end-----')

    if (Q3):
        print(f'-------Q3:------')
        print(coronaDataToVector('./Covid-19-USA.txt'))
        print(f'-----Q3 end-----')

    if (Q4):
        print(f'-------Q4:------')
        X = np.asarray([[1, 2],
                        [1, 2],
                        [1, 2]])
        # TODO Import  X
        # C = np.asarray([[1, 0],
        #                [0, 1]])
        # TODO Import
        w = np.asarray([[2],
                        [2],
                        [2]])

        nlen = 3
        mlen = 5
        X = np.asarray([[1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 0]])

        X = np.asarray([[1.6, 1, 1, 1.5, 8],
                        [1, 0.3, 2, 0, 3],
                        [1, 1, 1, 3, 5.6]])
        # TODO Import  X
        C = np.asarray([[1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1]])
        # TODO Import
        w = np.asarray([[0.6],
                        [2],
                        [3]])

        # TODO Import w
        section_a = [LRobjective(w, X, C), LRGradient(w, X, C), LRHessian(X, w)]
        f = lambda x: LRobjective(w,x,C)
        # TODO loop through different epsilons, and show the diff (for grad test)
        gradTest(f,X,section_a[1])
        print(f'-----Q4 end-----')
