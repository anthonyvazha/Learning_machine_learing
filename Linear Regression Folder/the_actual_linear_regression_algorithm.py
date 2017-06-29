from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('fivethirtyeight')

#random simple values,turing them into arrays and also being really explict with data types
#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

#plot the values in a scatter plot
# plt.scatter(xs,ys)
# plt.show()

#this class if meant for testing purposes(comment out the top values for testing)
def create_dataset(hm, variance, step=2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)


def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - (mean(xs * ys))) /
        (((mean(xs)**2)) - mean(xs**2)) )
    b = mean(ys)-(m*mean(xs))
    return m,b
#this just the distance squared
def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)
#this is the right side of the error equation
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig)for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean =  squared_error(ys_orig, y_mean_line)
    #final formula of error
    return 1 - (squared_error_regr / squared_error_y_mean)
#testing the best fit line
xs,ys = create_dataset(40,10, 2, correlation='pos')




m,b = best_fit_slope_and_intercept(xs,ys)
print(m,b)

regression_line = [(m*x) + b for x in xs]
#same as the line ^^^^
# for x in xs:
#     regression_line.append((m*x)+b)
predict_x = 8
predict_y = (m*predict_x)+b
#accuracy vs. confidence to find best fit lines
#squared error is how you deal with this problem
#distance between point and the best fit line squared
#we square it because to get positive values and get rid of outliers in the data
#it takes care of outliers & positive values
#you can also do power of 4, 6, 18 but 2 is the most common amoung industry
#SE = squared error, y_hat is reggresion line, y_meanline is horizontal straing line
#Foumula = r^2=1-(SE)y_hat/(SE)y_meanline
#we really want the r^2 really high to get better results
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, s = 100, color ='g')
plt.plot(xs, regression_line)
plt.show()
