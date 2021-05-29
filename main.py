import math
from datetime import time

import numpy as np
from scipy import sparse as sparse
import matplotlib.pyplot as plt
import numpy.linalg as alg

import loadMNIST


# import loadMNIST_V2

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
    return 1.0 / (1.0 + np.exp(matrix_mul) + 0.0000001)


def LRobjective(w, X, C):
    m = len(C[0])
    c1 = np.asarray([C[0]])
    c2 = np.asarray([C[1]])
    res = (-1 / m) * ((c1 @ np.log(sigmoid(X.T, w))) + (c2 @ np.log(1 - sigmoid(X.T, w))))
    return res[0]


def LRGradient(w, X, C):
    m = len(C[0])
    c1 = np.asarray([C[0]])
    res = (1 / m) * X @ (sigmoid(X.T, w) - c1.T)
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


def hessianTest(gradFunc, hessFunc, w, sampleVec):
    d = sampleVec
    numOfIter = 7
    epsilon = 0.1
    fx = gradFunc(w)
    hessVal = hessFunc(w)
    y_0 = np.zeros(numOfIter)
    y_1 = np.zeros(numOfIter)
    for k in range(1, numOfIter):
        epsilon = epsilon * pow(1 / 2, k)
        F_xd = gradFunc(w + epsilon * d)
        jacMV = hessVal @ (epsilon * d)
        y_0[k - 1] = np.linalg.norm(F_xd - fx)
        y_1[k - 1] = np.linalg.norm(F_xd - fx - jacMV)
    plt.figure()
    plt.semilogy([i for i in range(1, numOfIter + 1)], y_0)
    plt.semilogy([i for i in range(1, numOfIter + 1)], y_1)
    plt.title("Successful Jacbian test in semiLog plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.legend(["First order approx", "Second order approx"])
    plt.show()


def gradTest(f, w, gradX, sampleVec):
    d = sampleVec
    numOfIter = 8
    epsilon = 0.1
    f_0 = f(w)
    y_0 = np.zeros(numOfIter)
    y_1 = np.zeros(numOfIter)

    for k in range(1, numOfIter):
        epsilon = epsilon * pow(1 / 2, k)
        Fk = f(w + epsilon * d)
        F1 = f_0 + epsilon * (d.T @ gradX)
        y_0[k - 1] = abs(Fk - f_0)
        y_1[k - 1] = abs(Fk - F1)

    plt.figure()
    plt.semilogy([i for i in range(1, numOfIter + 1)], y_0)
    plt.semilogy([i for i in range(1, numOfIter + 1)], y_1)
    plt.title("Successful Grad test in semiLog plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.legend(["first order approx", "second order approx"])
    plt.show()


def load(numOftest, dig1, dig2):
    # valid input
    if numOftest > 60000:
        print(f'number of picture is {numOftest}>60000 or 0')
        return

    # loading data from MNIST
    loader = loadMNIST.MnistDataloader("./dataset/MNIST/train-images.idx3-ubyte",
                                       "./dataset/MNIST/train-labels.idx1-ubyte",
                                       "./dataset/MNIST/t10k-images.idx3-ubyte",
                                       "./dataset/MNIST/t10k-labels.idx1-ubyte")
    imageData = np.asarray(loader.load_data()[0][0])
    imageLabels = np.asarray(loader.load_data()[0][1])

    # flatten the vector
    imageData = imageData.reshape(-1, 784)

    # normalizing the vector
    imageData = imageData / 255

    # clearing the data only for our digits
    data = []
    labels = []
    test = []
    i = 0
    chosenCounter = 0

    while i < len(imageData) and chosenCounter < numOftest:
        if imageLabels[i] == dig1 or imageLabels[i] == dig2:
            data.append(imageData[i])
            labels.append(imageLabels[i])
            chosenCounter += 1
        i += 1

    while True:
        if imageLabels[i] == dig1 or imageLabels[i] == dig2:
            test.append(imageData[i])
            break
        i += 1

    # create C as binary labels
    label1 = np.asarray(labels)
    label2 = np.asarray(labels)
    for i in range(0, numOftest):
        if labels[i] == dig1:
            label1[i] = 1
            label2[i] = 0
        else:
            label1[i] = 0
            label2[i] = 1

    C = np.asarray([label1, label2])
    return np.asarray(data).T, C, np.asarray(test).T


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
        X, C, w = load(100, 0, 1)
        print(f'loaded DATA shape:{X.shape}')
        #  print(f'labels are:\n{C}')
        print(f'w shape:{w.shape}')
        m = len(C[0])
        section_a = [LRobjective(w, X, C), LRGradient(w, X, C), LRHessian(X, w, m)]
        f = lambda w1: LRobjective(w1, X, C)
        d = unitVectorGenerator(len(X))
        gradTest(f, w, section_a[1], d)
        gradFunc = lambda w2: LRGradient(w2, X, C)
        hessFunc = lambda w_3: LRHessian(X, w_3, m)
        hessianTest(gradFunc, hessFunc, w, d)
        print(f'-----Q4 end-----')
