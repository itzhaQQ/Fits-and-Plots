import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.odr import *


# -------------------Geradenfit--------------------
def geradenfit(x, y, x_err, y_err):
    print('---------------Geradenfit----------------')
    # Mittelwert
    def mittel(x, n):
        return (1 / n) * np.sum(x)

    # varianzgewichteter Mittelwert
    def mittel_var(val, z):
        return np.sum(z / (val ** 2)) / np.sum(1 / (val ** 2))

    # varianzgemittelte Standardabweichung
    def sigma(val, n):
        return n / (np.sum(1 / val ** 2))

    # gerade
    def polynom(m, b, x):
        return m * x + b

    if len(x)==len(y):
        n = len(x)
        print('List Works')
    else:
        print('x and y are not the same length')

    x_strich = mittel(x, n)
    x2_strich = mittel(x ** 2, n)
    y_strich = mittel(y, n)
    xy_strich = mittel(x * y, n)
    print(f'{x_strich = }')
    print(f'{y_strich = }')
    print(f'{x2_strich =}')
    print(f'{xy_strich = }')
    print('----------------------------------------------------------------------------')
    m = (xy_strich - (x_strich * y_strich)) / (x2_strich - x_strich ** 2)
    b = (x2_strich * y_strich - x_strich * xy_strich) / (x2_strich - x_strich ** 2)
    print(f'Steigung: {m = }')
    print(f'y-Achsenabschnitt: {b = }')

    sigmax = sigma(x_err, n)
    sigmay = sigma(y_err, n)

    dm = np.sqrt(sigmay / (n * (x2_strich - x_strich ** 2)))
    db = np.sqrt(sigmay * x2_strich / (n * (x2_strich - (x_strich ** 2))))
    print(f'Fehler Steigung: {dm = }')
    print(f'Fehler y-Achsenabschnitt {db = }')
    print('----------------------------------------------------------------------------')
    # create dictionary for further calculations
    dict = {
        'm':m,
        'b':b,
        'dm':dm,
        'db':db,
    }
    return dict

    # plot
    print('----------------------------------------------------------------------------')
    fig, ax = plt.subplots()
    ax.set_title('Name')
    ax.set_ylabel(r'y-Achse')
    ax.set_xlabel(r'x-Achse')
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, capsize=2, fmt='.', markersize=5, color='black')
    ax.plot(x, polynom(m,b,x), label ='label1')
    ax.plot(x, m * x + b, label=f'$y = ({m:0.3e})x+({b:+0.3e})$')
    plt.legend()
    plt.grid(visible=True, which='major', color='666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='999999', linestyle='-', alpha=0.5)
    plt.show()
# geradenfit(x-wert, y-wert, x-fehler, y-fehler, länge)

# --------------multiple_Geradenfit----------------

def multifit(x, y, x_err, y_err):
    print('---------------Geradenfit----------------')
 
    # Mittelwert
    def mittel(x, n):
        return (1 / n) * np.sum(x)

    # varianzgewichteter Mittelwert
    def mittel_var(val, z):
        return np.sum(z / (val ** 2)) / np.sum(1 / (val ** 2))

    # varianzgemittelte Standardabweichung
    def sigma(val, n):
        return n / (np.sum(1 / val ** 2))

    # gerade
    def polynom(m, b, x):
        return m * x + b

    if len(x)==len(y):
        n = len(x)
        print('List Works')
    else:
        print('x and y are not the same length')

    x_strich = mittel(x, n)
    x2_strich = mittel(x ** 2, n)
    y_strich = mittel(y, n)
    xy_strich = mittel(x * y, n)
    print(f'{x_strich = }')
    print(f'{y_strich = }')
    print(f'{x2_strich =}')
    print(f'{xy_strich = }')
    print('----------------------------------------------------------------------------')
    m = (xy_strich - (x_strich * y_strich)) / (x2_strich - x_strich ** 2)
    b = (x2_strich * y_strich - x_strich * xy_strich) / (x2_strich - x_strich ** 2)
    print(f'Steigung: {m = }')
    print(f'y-Achsenabschnitt: {b = }')

    sigmax = sigma(x_err, n)
    sigmay = sigma(y_err, n)

    dm = np.sqrt(sigmay / (n * (x2_strich - x_strich ** 2)))
    db = np.sqrt(sigmay * x2_strich / (n * (x2_strich - (x_strich ** 2))))
    print(f'Fehler Steigung: {dm = }')
    print(f'Fehler y-Achsenabschnitt {db = }')
    print('----------------------------------------------------------------------------')
    # create dictionary for further calculations
    dict = {
        'm':m,
        'b':b,
        'dm':dm,
        'db':db,
    }
    return dict

"""

example for mutiple fits:

result_1 = geradenfit_data(x_1, y_1, x_err_1, y_err_1)
result_2 = geradenfit_data(x_2, y_2, x_err_2, y_err_2)
[...]
result_n = geradenfit_data(x_n, y_n, x_err_n, y_err_n)

"""
# Plot
"""

dictarray = np.array([result_1, result_2, result_n])    # array for dictionaries of the fits:

# define arrays for the collected data:

x_arrays = np.array([x_1, x_2, x_n], dtype=object)
x_err_arrays = np.array([x_err_1, x_err_2, x_err_n], dtype=object)
y_arrays = np.array([y_1, y_2, y_n], dtype=object)
y_err_arrays = np.array([y_err_1, y_err_2, y_err_n], dtype=object)

colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])   # array with colors

"""

def multifit_plot(x_array, y_array, x_err_array, y_err_array, dictarray):
    # plot
    # use r'' to be able to use LaTeX Code, LaTeX Code needs to be in $$
    fig, ax = plt.subplots(dpi=500)

    # general title and axis labels
    ax.set_title('Titel XY')
    ax.set_xlabel(r'Name $[Einheit]$')
    ax.set_ylabel(r'$\lambda$ $[mm]$')

    # the data points and fits
    for i in range(0, len(dictarray)):
        ax.errorbar(x_array[i], y_array[i], xerr=x_err_array[i], yerr=y_err_array[i], capsize=2., linestyle='',
                    marker='o', markersize=2, label='Messwerte', color=colors[i])
        ax.plot(x_array[i], dictarray[i]["m"] * x_array[i] + dictarray[i]["b"], label='Fit', color=colors[i])

    # code for a nice grid
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.5)

    # include a legend
    plt.legend()
    plt.show()
    # save plot as a file
    # plt.savefig('Titel.jpg')
    print('----------------------------------------------------------------------------')
# geradenfit_plot(x_array, y_array, x_err_array, y_err_array, dictarray)


# -------------------Cosinusfit--------------------
def cosinusfit(x, y, x_err, y_err):

    # get parameter a, b, c
    def function(x, a, b, c):
        return a*(np.cos(x+b)**2) + c

    param, cov = curve_fit(function, x, y)
    print("Parameter des Cosinus-Fit:\n\na: " + str(param[0]) + "\nb: " + str(param[1])+ "\nc: " + str(param[2]))

    # Define Function for our data
    def cos_function(p, x):
        c, a, v = p
        return v * np.cos(x - a) ** 2 + c

    # Create model for storing information about function
    cos_model = Model(cos_function)

    # Create RealData-object with our data
    data = RealData(x, y, sx=x_err, sy=y_err)

    # Set up ODR(Orthogonal Distance Regression) with the model and data
    # beta0 for input of guessed parameters
    odr = ODR(data, cos_model, beta0=[20, 0, 120])

    # Run regression
    out = odr.run()

    # In-built print mehthod to give results

    out.pprint()

    # Plot
    x_fit = np.linspace(x[0], x[-1], 1000)
    y_fit = cos_function(out.beta, x_fit)

    fig, ax = plt.subplots()
    ax.set_title('Title')
    ax.set_ylabel(r'y-Label')
    ax.set_xlabel(r'x-Label')
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, capsize=2, fmt='.', markersize=5, color='black', label='Messpunkte')
    plt.plot(x_fit, y_fit, label='Fit')
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.legend()
    plt.show()
# cosinusfit(x, y, x_err, y_err)


