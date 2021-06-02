import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as alg
from scipy import sparse as sparse

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


def sigmoid(X, w):
    XT = X.T
    matrix_mul = -1 * XT @ w
    res = np.asarray(1.0 / (1.0 + np.exp(matrix_mul)))
    return res


def LRobjective(w, X, C):
    m = len(C[0])
    c1 = np.asarray([C[0]])
    c2 = np.asarray([C[1]])
    res = (-1 / m) * ((c1 @ np.log(sigmoid(X, w))) + (c2 @ np.log(1 - sigmoid(X, w))))
    return res


def LRGradient(w, X, C):
    m = len(C[0])
    c1 = np.asarray(C[0])
    c1 = c1.reshape((c1.shape[0], 1))
    sig = (sigmoid(X, w) - c1) ##TODO this is a matrix. should be vector
    res = (1 / m) * X @ sig
    return res


def unitVectorGenerator(vecSize):
    v = np.asarray([np.random.randn(vecSize)])
    normalized_V = v / np.linalg.norm(v)
    return normalized_V.T


def LRHessian(X, w, m):
    elementWise = np.multiply(sigmoid(X, w), (1 - sigmoid(X, w)))
    D = np.zeros((len(elementWise), len(elementWise)))
    for i in range(len(elementWise)):
        D[i][i] = elementWise[i]
    return (1 / m) * X @ D @ X.T


def hessianTest(gradFunc, hessFunc, w, sampleVec):
    d = sampleVec
    numOfIter = 7
    epsilon = 1
    fx = gradFunc(w)
    hessVal = hessFunc(w)
    y_0 = []
    y_1 = []
    eps = []
    for k in range(1, numOfIter):
        F_xd = gradFunc(w + epsilon * d)
        jacMV = hessVal @ (epsilon * d)
        y_0.append(np.linalg.norm(F_xd - fx, 2))
        y_1.append(np.linalg.norm(F_xd - fx - jacMV, 2))
        eps.append(epsilon)
        epsilon = epsilon * 1 / 2

    plt.figure()
    for i in range (len(y_0)):
        print (y_1[i]/y_0[i])
    plt.plot(eps, y_0)
    plt.plot(eps, y_1)
    plt.title("Successful Jacbian test in semiLog plot")
    plt.xlabel("epsilon")
    plt.ylabel("error")
    plt.legend(["First order approx", "Second order approx"])
    plt.gca().invert_xaxis()

    plt.show()


def gradTest(f, w, gradX, sampleVec):
    d = sampleVec
    numOfIter = 8
    epsilon = 1
    f_0 = f(w)
    y_0 = []
    y_1 = []
    eps = []
    for k in range(1, numOfIter):
        Fk = f(w + epsilon * d)
        F1 = f_0 + epsilon * (d.T @ gradX)
        first = np.abs(Fk - f_0)
        second = np.abs(Fk - F1)
        y_0.append(np.abs(Fk - f_0))
        y_1.append(np.abs(Fk - F1))
        eps.append(epsilon)
        epsilon = epsilon * 1/2

    plt.figure()
    openy0 = []
    openy1 = []

    for i in range (len(y_0)):
        openy0.append( y_0[i][0][0])
        openy1 .append(y_1 [i][0][0])
    plt.semilogy(eps, openy0)
    plt.semilogy(eps, openy1)
    plt.title("Successful Grad test in semiLog plot")
    plt.xlabel("epsilon")
    plt.ylabel("error")
    plt.legend(["first order approx", "second order approx"])
    plt.gca().invert_xaxis()
    plt.show()


def load(numToTrainOn, dig1, dig2):
    limit = 30000
    # valid input
    if numToTrainOn > 60000 or numToTrainOn > limit:
        print(f'number of picture is {numToTrainOn}>60000 or 0')
        return

    # loading data from MNIST
    loader = loadMNIST.MnistDataloader("./dataset/MNIST/train-images.idx3-ubyte",
                                       "./dataset/MNIST/train-labels.idx1-ubyte",
                                       "./dataset/MNIST/t10k-images.idx3-ubyte",
                                       "./dataset/MNIST/t10k-labels.idx1-ubyte")
    imageData = np.asarray(loader.load_data()[0][0])
    imageLabels = np.asarray(loader.load_data()[0][1])
    testData = np.asarray(loader.load_data()[1][0])
    testLabels = np.asarray(loader.load_data()[1][1])

    # flatten the vector
    imageData = imageData.reshape(-1, 784)
    testData = testData.reshape(-1, 784)

    # normalizing the vector
    imageData = imageData / 255
    testData = testData / 255

    # clearing the data only for our digits
    # strating with train data
    FilteredTrainImages = []
    FilteredTrainLabels = []
    FilteredTestImages = []
    FilteredTestLabels = []

    i = 0
    chosenCounter = 0
    while i < limit and chosenCounter < numToTrainOn:
        if imageLabels[i] == dig1 or imageLabels[i] == dig2:
            FilteredTrainImages.append(imageData[i])
            FilteredTrainLabels.append(imageLabels[i])
            chosenCounter += 1
        i += 1

    # moving to test data
    i = 0
    while i < len(testData):
        if testLabels[i] == dig1 or testLabels[i] == dig2:
            FilteredTestImages.append(testData[i])
            FilteredTestLabels.append(testLabels[i])
        i += 1

    # create C as binary labels
    label1 = np.asarray(FilteredTrainLabels)
    label2 = np.asarray(FilteredTrainLabels)
    for i in range(0, len(FilteredTrainLabels) - 1):
        if FilteredTrainLabels[i] == dig1:
            label1[i] = 1
            label2[i] = 0
        else:
            label1[i] = 0
            label2[i] = 1
    C = np.asarray([label1, label2])

    # creating verification
    testLabel1 = np.asarray(FilteredTestLabels)
    testLabel2 = np.asarray(FilteredTestLabels)
    for i in range(0, len(FilteredTestLabels)):
        if FilteredTestLabels[i] == dig1:
            testLabel1[i] = 1
            testLabel2[i] = 0
        else:
            testLabel1[i] = 0
            testLabel2[i] = 1
    results = np.asarray([testLabel1, testLabel2])

    return np.asarray(FilteredTrainImages).T, C, np.asarray(FilteredTestImages).T, results


def SD(trainData, trainLabels, testImages, testLabels, one_zero):
    trainSamples = []
    testSamples = []
    weight = np.asarray([[0]] * 784)
    c1 = np.asarray([trainLabels[0]])
    c2 = np.asarray([trainLabels[1]])
    first = True
    iter = []
    for i in range(100):
        iter.append(i)
        # calculating for train
        # print("calculating for train")
        prevW = weight
        direction = -1 * LRGradient(prevW, trainData, trainLabels)
        fFunc = lambda w: LRobjective(w, trainData, trainLabels)
        alpha = armijo(prevW, fFunc, -1 * direction, direction)
        weight = np.clip(prevW + alpha * direction, -1, 1)

        # calculating for test
        trainSamples.append(LRobjective(weight, trainData, trainLabels))
        testSamples.append(LRobjective(weight, testImages, testLabels))

        if ((not first) and (testConverge(weight, prevW))):
            break
        first = False

    f_train_diffs = []
    f_test_diffs = []
    n = len(testSamples)
    for i in range(n):
        f_test_diffs.append(np.abs(testSamples[i] - testSamples[n - 1])[0])
        f_train_diffs.append(np.abs(trainSamples[i] - trainSamples[n - 1]))

    testDiff = []
    trainDiff = []

    for i in range(len(f_test_diffs)):
        testDiff.append(f_test_diffs[i][0])
        trainDiff.append(f_train_diffs[i][0][0])

    plt.figure()
    title = ""
    if (one_zero):
        title = "1/0, |f(w)-f(w*)|"
    else:
        title = "8/9, |f(w)-f(w*)|"
    plt.semilogy(iter, testDiff)
    plt.semilogy(iter, trainDiff)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.legend(["test Difference", "train Difference"])
    plt.show()

    # TODO Plot both diffs on the same graph. x label is "iterations", y label is "objective value"

    return weight


def newton(trainData, trainLabels, testImages, testLabels, one_zero):
    trainSamples = []
    testSamples = []
    weight = np.asarray([[0]] * 784)
    c1 = np.asarray([trainLabels[0]])
    c2 = np.asarray([trainLabels[1]])
    m = len(testLabels[0])
    iter = []
    first = True
    for i in range(100):
        iter.append(i)
        # calculating for train
        # print("calculating for train")
        prevW = weight
        hess = LRHessian(trainData, prevW, m)
        gradF = LRGradient(prevW, trainData, trainLabels)
        try:
            direction = -1 * np.linalg.inv(hess) @ gradF
        except:
            hess = hess + (0.1 * np.eye(len(hess)))
            direction = -1 * np.linalg.inv(hess) @ gradF

        direction = -1 * np.linalg.inv(hess) @ gradF
        fFunc = lambda w: LRobjective(w, trainData, trainLabels)
        alpha = armijo(prevW, fFunc, gradF, direction)
        weight = np.clip(prevW + alpha * direction, -1, 1)

        # calculating for test
        trainSamples.append(LRobjective(weight, trainData, trainLabels))
        testSamples.append(LRobjective(weight, testImages, testLabels))

        if ((not first) and (testConverge(weight, prevW))):
            break
        first = False

    f_train_diffs = []
    f_test_diffs = []
    n = len(testSamples)
    for i in range(n):
        f_test_diffs.append(np.abs(testSamples[i] - testSamples[n - 1])[0])
        f_train_diffs.append(np.abs(trainSamples[i] - trainSamples[n - 1]))

    testDiff = []
    trainDiff = []

    for i in range(len(f_test_diffs)):
        testDiff.append(f_test_diffs[i][0])
        trainDiff.append(f_train_diffs[i][0][0])

    title = ""
    if (one_zero):
        title = "1/0, |f(w)-f(w*)|"
    else:
        title = "8/9, |f(w)-f(w*)|"
    plt.semilogy(iter, testDiff)
    plt.semilogy(iter, trainDiff)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.legend(["test Difference", "train Difference"])
    plt.show()

    # TODO Plot both diffs on the same graph. x label is "iterations", y label is "objective value"

    return weight


def armijo(weight, objectiveF, gradF, direction):
    alpha = 1
    beta = 1 / 2
    c = 1 * pow(10, -4)
    f_x = objectiveF(weight)

    for i in range(10):
        f_alpha = objectiveF(weight + alpha * direction)
        if (f_alpha <= f_x + c * alpha * np.dot(gradF.T[0], direction.T[0])):
            return alpha
        else:
            alpha = beta * alpha

    return alpha

def eranTest(f, w, sampleVec):
    d = sampleVec
    numOfIter = 8
    epsilon = 1
    gradX = w
    f_0 = f(w)
    y_0 = []
    y_1 = []
    eps = []
    iter = []
    for k in range(numOfIter):
        Fk = f(w + epsilon * d)
        F1 = f_0 + epsilon * np.dot(np.reshape(d,len(d)) , np.reshape(gradX,len(gradX)))
        first = np.abs(Fk - f_0)
        second = np.abs(Fk - F1)
        y_0.append(np.abs(Fk - f_0))
        y_1.append(np.abs(Fk - F1))
        eps.append(epsilon)
        epsilon = epsilon * pow(1/2,k)
        iter.append(k)
    plt.figure()
    openy0 = []
    openy1 = []

    # for i in range(len(y_0)):
    #     openy0.append(y_0[i][0][0])
    #     openy1.append(y_1[i][0][0])
    plt.semilogy(iter, y_0)
    plt.semilogy(iter, y_1)
    plt.title("eran test")
    plt.xlabel("epsilon")
    plt.ylabel("error")
    plt.legend(["first order approx", "second order approx"])
    # plt.gca().invert_xaxis()
    plt.show()



def eranFunc(x):
    x = np.reshape(x, 784)
    return 1/2 * np.dot(x,x)

def testConverge(weight, prevW):
    normsOutput = np.linalg.norm(weight - prevW) / np.linalg.norm(prevW)
    return normsOutput < 0.001


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
        plt.title("Original F")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # our part
        xLS = Q2a(y, G)
        xIRLS = IRLS(f, x, y, G, 10, 1, 0.001)
        plt.figure()
        plt.plot(xLS)
        plt.plot(xIRLS)
        plt.title("Approx F")
        plt.legend(["LS", "IRLS"])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        print(f'-----Q2 end-----')

    if (Q3):
        print(f'-------Q3:------')
        print(coronaDataToVector('./Covid-19-USA.txt'))
        print(f'-----Q3 end-----')

    if (Q4):
        print(f'-------Q4:------')
        testImages, testLabels, trainData, trainLabels = load(30000, 8, 9)

        ##############################################################
        # test section, delete when done

        #############################################################
        # TODO make conclutions

        print(f'loaded DATA shape:{testImages.shape}')
        print(f'loaded DATA shape:{trainData.shape}')
        print(f'loaded DATA labels shape:{testLabels.shape}')
        print(f'results are:{trainLabels.shape}')
        m = len(testLabels[0])
        f = lambda w1: LRobjective(w1, testImages, testLabels)
        d = unitVectorGenerator(len(testImages))
        w = np.clip(unitVectorGenerator(len(testImages)), -1, 1)
        #############################################
        # leave like this, ill use it later
        gradFunc = lambda w2: LRGradient(w2, testImages, testLabels)
        hessFunc = lambda w_3: LRHessian(testImages, w_3, m)
        hessianTest(gradFunc, hessFunc, w, d)
        gradTest(f, w, gradFunc(w), d)
        # eranTest(eranFunc, w, d)
        ##############################################
        SD(trainData, trainLabels, testImages, testLabels , False)
        newton(trainData, trainLabels, testImages, testLabels, False)

        print(f'-----Q4 end-----')
