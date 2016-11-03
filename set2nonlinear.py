import numpy as np
import random
# import math
# import matplotlib.pyplot as plt


class NonLineRegress:
    ''' Class for the Linear Regression
    '''

    def __init__(self, n, noise, transform):
        ''' This linear regress. uses the given target function and generates
            n random data points x_n from [-1, 1] x [-1, 1]. The outputs y_n
            are stored into the list 'y'. The weight vector w is
            initialized to the pseudo-inverse(x) * y. If the boolean
            'transform' is specified true, the lin regress will perfrom the
            given transform on the x_n's.
        '''
        self.x = np.array([(1.0, np.random.uniform(-1.0, 1.0),
                          np.random.uniform(-1.0, 1.0)) for i in range(n)])
        self.y = np.array([int(np.sign(x[1]*x[1] + x[2]*x[2] - 0.6))
                          for x in self.x])
        if (transform):
            newx = []
            for element in self.x:
                temp = (element[0], element[1], element[2],
                        element[1] * element[2], element[1] * element[1],
                        element[2] * element[2])
                newx.append(temp)
            self.x = np.array(newx)

        self.addNoise(noise)
        self.inverse = np.linalg.pinv(self.x)
        self.w = self.inverse.dot(self.y)

    def addNoise(self, decimal):
        for i in range(int(decimal * 1000)):
            rand = random.randint(0, 999)
            self.y[rand] = self.y[rand] * -1


def getErrorIn(n, points, noise, transform):
    ''' Gets the average Error-In for n runs of the line regress with the
        specified noise, number of points, and transform boolean.
    '''
    total = 0.0
    for i in range(n):
        lr = NonLineRegress(points, noise, transform)
        wy = np.array([int(np.sign(lr.w.transpose().dot(x)))
                       for x in lr.x])
        wrong = 0.0
        for i, element in enumerate(wy):
            if element != lr.y[i]:
                wrong += 1.0
        total += (wrong / points)
    print (total / n)
    return total / n


def getErrorOut(n, points, noise, transform):
    ''' Gets the average Error-out for n runs of the line regress with the
        specified noise, number of points, and transform boolean.
    '''
    total = 0.0
    for i in range(n):
        lr = NonLineRegress(points, noise, transform)
        newData = np.array([(1.0, np.random.uniform(-1.0, 1.0),
                            np.random.uniform(-1.0, 1.0))
                            for i in range(points)])
        original = np.array([int(np.sign(x[1]*x[1] + x[2]*x[2] - 0.6))
                             for x in newData])
        newx = []
        for element in newData:
            temp = (element[0], element[1], element[2],
                    element[1] * element[2], element[1] * element[1],
                    element[2] * element[2])
            newx.append(temp)
        newData = np.array(newx)

        for i in range(int(noise * points)):
            rand = random.randint(0, 999)
            original[rand] = original[rand] * -1

        wy = np.array([int(np.sign(lr.w.transpose().dot(x)))
                       for x in newData])
        wrong = 0.0
        for i, element in enumerate(wy):
            if element != original[i]:
                wrong += 1.0
        total += (wrong / 1000)
    print (total / n)
    return (total / n)


def getBestW(n, points, noise, transform):
    ''' get best w~ from answer choices [A, B, C, D, E] by comparing the output
        of w and the given choice's w~.
    '''
    total = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(n):
        lr = NonLineRegress(points, noise, transform)
        newData = np.array([(1.0, np.random.uniform(-1.0, 1.0),
                            np.random.uniform(-1.0, 1.0))
                            for i in range(points)])
        gA = np.array([-1.0, -0.05, 0.08, 0.13, 1.5, 1.5])
        gB = np.array([-1.0, -0.05, 0.08, 0.13, 1.5, 15.0])
        gC = np.array([-1.0, -0.05, 0.08, 0.13, 15.0, 1.5])
        gD = np.array([-1.0, -1.5, 0.08, 0.13, 0.05, 0.05])
        gE = np.array([-1.0, -0.05, 0.08, 1.5, 0.15, 0.15])

        newx = []
        for element in newData:
            temp = (element[0], element[1], element[2],
                    element[1] * element[2], element[1] * element[1],
                    element[2] * element[2])
            newx.append(temp)
        newData = np.array(newx)

        wy = np.array([int(np.sign(lr.w.transpose().dot(x)))
                       for x in newData])
        wA = np.array([int(np.sign(gA.transpose().dot(x)))
                       for x in newData])
        wB = np.array([int(np.sign(gB.transpose().dot(x)))
                       for x in newData])
        wC = np.array([int(np.sign(gC.transpose().dot(x)))
                       for x in newData])
        wD = np.array([int(np.sign(gD.transpose().dot(x)))
                       for x in newData])
        wE = np.array([int(np.sign(gE.transpose().dot(x)))
                       for x in newData])

        wrong = [0.0, 0.0, 0.0, 0.0, 0.0]
        for i, element in enumerate(wy):
            if element != wA[i]:
                wrong[0] += 1.0
            if element != wB[i]:
                wrong[1] += 1.0
            if element != wC[i]:
                wrong[2] += 1.0
            if element != wD[i]:
                wrong[3] += 1.0
            if element != wE[i]:
                wrong[4] += 1.0
        for i, r in enumerate(total):
            total[i] += (wrong[i] / 1000)

    for i, element in enumerate(total):
        total[i] = total[i] / 1000.0
    print (total)
    return (total)


print("Error-In:")
getErrorIn(1000, 1000, 0.1, False)
print("Get errors for answer choices [A, B, C, D, E]:")
getBestW(1000, 1000, 0.1, True)
print("Error-Out:")
getErrorOut(1000, 50, 0.1, True)
