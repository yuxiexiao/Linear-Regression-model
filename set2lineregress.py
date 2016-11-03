import numpy as np
import random
# import math
# import matplotlib.pyplot as plt


class LineRegress:
    ''' Class for the Linear Regression
    '''

    def __init__(self, n, noise):
        ''' Generates a random target function f, and generates
            n random data points x_n from [-1, 1] x [-1, 1]. The outputs y_n
            are stored into the list 'y'. The weight vector w is
            initialized to the pseudo-inverse(x) * y. Also adds noise to a
            specified fraction of the dataset.
        '''
        x1, x2, y1, y2 = [np.random.uniform(-1.0, 1.0) for i in range(4)]
        self.target = np.array([x2*y1 - x1*y2, y2-y1, x1-x2])
        self.x = np.array([(1.0, np.random.uniform(-1.0, 1.0),
                          np.random.uniform(-1.0, 1.0)) for i in range(n)])
        self.y = np.array([int(np.sign(self.target.transpose().dot(x)))
                           for x in self.x])
        self.addNoise(noise)
        self.inverse = np.linalg.pinv(self.x)
        self.w = self.inverse.dot(self.y)

    def addNoise(self, decimal):
        for i in range(int(decimal * 1000)):
            rand = random.randint(0, 999)
            self.y[rand] = self.y[rand] * -1


def getErrorIn(n, points, noise):
    ''' Generates the average Error-In for n runs of a line regression with
        the specified noise and for the specified number of points per run.
    '''

    total = 0.0
    for i in range(n):
        lr = LineRegress(points, noise)
        wy = np.array([int(np.sign(lr.w.transpose().dot(x)))
                       for x in lr.x])
        wrong = 0.0
        for i, element in enumerate(wy):
            if element != lr.y[i]:
                wrong += 1.0
        total += (wrong / points)
    print (total / n)
    return total / n


def getErrorOut(n, points, noise):
    ''' Generates the average Error-out for n runs of a line regression with
        the specified noise and for the specified number of points per run.
    '''

    total = 0.0
    for i in range(n):
        lr = LineRegress(points, noise)
        newData = np.array([(1.0, np.random.uniform(-1.0, 1.0),
                            np.random.uniform(-1.0, 1.0))
                            for i in range(1000)])
        wy = np.array([int(np.sign(lr.w.transpose().dot(x)))
                       for x in newData])
        original = np.array([int(np.sign(lr.target.transpose().dot(x)))
                             for x in newData])
        wrong = 0.0
        for i, element in enumerate(wy):
            if element != original[i]:
                wrong += 1.0
        total += (wrong / 1000)
    print (total / n)
    return (total / n)


class PLA:
    ''' Class for the Perceptron Learning Algorithm
    '''

    def __init__(self, n):
        ''' Generates a random target function f, and generates
            n random data points x_n from [-1, 1] x [-1, 1]. The outputs y_n
            are stored into the list 'y'. The weight vector w is
            initialized to the pseudo-inverse(x) * y.
        '''
        x1, x2, y1, y2 = [np.random.uniform(-1.0, 1.0) for i in range(4)]
        self.target = np.array([x2*y1 - x1*y2, y2-y1, x1-x2])
        self.x = np.array([(1.0, np.random.uniform(-1.0, 1.0),
                          np.random.uniform(-1.0, 1.0)) for i in range(n)])
        self.y = np.array([int(np.sign(self.target.transpose().dot(x)))
                           for x in self.x])
        self.inverse = np.linalg.pinv(self.x)
        self.w = self.inverse.dot(self.y)

    def runPLA(self):
        ''' Runs the pla to generate w that estimates f
        '''
        wy = np.array([int(np.sign(self.w.transpose().dot(x)))
                      for x in self.x])  # signs generated from w

        misclass = []
        # get the misclassified points
        for i, element in enumerate(wy):
            if (element != self.y[i]):
                misclass.append([i, self.x[i]])

        iterations = 0
        while misclass:
            length = len(misclass)
            randIndex = random.randint(0, length - 1)
            index = misclass[randIndex][0]
            self.w += self.y[index] * misclass[randIndex][1]  # update w
            wy = [int(np.sign(self.w.transpose().dot(x)))
                  for x in self.x]
            misclass = []  # update misclassified points again
            for i, element in enumerate(wy):
                if (element != self.y[i]):
                    misclass.append([i, self.x[i]])
            iterations += 1

        return iterations


def getAverageIter(n, points):
    ''' Get the average number of iterations for n runs of the pla
        with N = points
    '''
    total = 0.0
    for i in range(n):
        pla = PLA(points)
        total += pla.runPLA()
    average = total / n
    print(average)
    return average


print("Ein:")
getErrorIn(1000, 100, 0)
print("Eout:")
getErrorOut(1000, 100, 0)
print("Average iterations of pla with linear regression:")
getAverageIter(1000, 10)



