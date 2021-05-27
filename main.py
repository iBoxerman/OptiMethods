import numpy as np
from scipy import sparse as sparse
import matplotlib.pyplot as plt
import numpy.linalg as alg


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


def LRsigmuind(XT, w):
    matrix_mul = -XT @ w
    return 1 / (1 + np.exp(matrix_mul))


def LRobjective(w, X, C):
    m = len(C[0])
    c1 = C[0].T
    c2 = C[1].T
    res = (-1 / m) * ((c1.T * np.log(LRsigmuind(X.T, w))) + (c2.T * np.log(1 - LRsigmuind(X.T, w))))
    return res


def LRGradient(w, X, C):
    m = len(C[0])
    c1 = C[0].T

    print(f'is this vector?:{LRsigmuind(X.T, w)}\n{c1}')
    res = (1 / m) * X @ (LRsigmuind(X.T, w) - c1)
    return res


def unitVectorGenerator(vecSize):
    v = np.random.randn(vecSize)
    normalized_V = v / np.linalg.norm(v)
    return normalized_V


def LRHessian(X, w):
    elementWise = np.multiply(LRsigmuind(X.T, w), (1 - LRsigmuind(X.T, w)))
    m = len(X[0])
    D = np.zeros((len(elementWise), len(elementWise)))
    for i in range(len(elementWise)):
        D[i][i] = elementWise[i]
    return (1 / m) * X @ D @ X.T

def gradTest(f,X, gradX):
    d = unitVectorGenerator(len(X))
    epsilons = [0.001, 0.00000001]
    Oone = lambda epsilon: abs(f(X + epsilon * d) - f(X))
    Otwo = lambda epsilon: abs( f(X + epsilon * d) - f(X) - epsilon * d.T @ gradX)
    plt.figure()
    plt.plot(Oone(epsilons[0]),Oone(epsilons[1]))
    plt.plot(Otwo(epsilons[1]),Otwo(epsilons[1]))
    plt.show()

def hessianTest(gradFunc, hesFunc, x):
    print("")

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
        C = np.asarray([[1, 1, 0],
                        [0, 0, 1]])
        # TODO Import
        w = np.asarray([[2],
                        [2],
                        [2]])
        # TODO Import w
        section_a = [LRobjective(w, X, C), LRGradient(w, X, C), LRHessian(X, w)]
        f = lambda x: LRobjective(w,x,C)
        # TODO loop through different epsilons, and show the diff (for grad test)
        gradTest(f,X,section_a[1])
        print(f'-----Q4 end-----')
