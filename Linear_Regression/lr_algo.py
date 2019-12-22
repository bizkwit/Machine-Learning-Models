import random
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)


def create_dataset(npoint, variance, step=2, correlation=False):
    '''
    A function that creates a data set for testing given parameters
    :param npoint: number of points the user wants to be in the data set
    :param variance: how variable does the user want the data set to be
    :param step: default=2, how far on average to step up or down per value
    :param correlation: positive or negative correlation between entries in the data set. depending on the step parameter
    :return: a data set of xs and ys
    '''
    val =1
    ys = []
    for i in range(npoint):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'positive':
            val += step
        elif correlation and correlation == 'negative':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    '''
    A function that finds the best fit slope and the y-intercept given the data
    :param xs: the data set
    :param ys: the labels
    :return: m=slope, b=y-intercept
    '''
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - (m * mean(xs))
    return m, b


def squared_error(yline, ypoints):
    '''
    A function that calculates the squared error between a point and its counterpart on the best fit line
    :param yline: y points on the line itself
    :param ypoints: the original y points
    :return: r^2
    '''
    return sum((yline-ypoints)**2)


def coefficient_of_determination(ypoints, yline):
    '''
    A function that finds the error rate
    :param ypoints: the original y points
    :param yline: y points on the line itself
    :return: the error rate (1 - (SE of best fit y) / (SE mean of y))
    '''
    y_mean_line = [mean(ypoints) for y in ypoints]
    squared_error_regression = squared_error(yline, ypoints)
    squared_error_mean = squared_error(y_mean_line, ypoints)
    return 1 - (squared_error_regression/squared_error_mean)


xs, ys = create_dataset(40, 30, 2, correlation="positive")


m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='r', s=100)
plt.plot(xs, regression_line)
plt.show()



