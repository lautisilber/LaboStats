import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chisquare, chi2
import matplotlib.pyplot as plt

class Model:
    def __init__(self, x, y, errx=None, erry=None):
        '''
            Initialize the model with the data.
            x: array of x values
            y: array of y values
        '''
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        if errx is None:
            self.errx = np.zeros(x.shape[0], dtype=float)
            self.errx.fill(np.finfo(float).eps)
        else:
            self.errx = np.array(errx, dtype=float)
        if erry is None:
            self.erry = np.zeros(y.shape[0], dtype=float)
            self.erry.fill(np.finfo(float).eps)
        else:
            self.erry = np.array(erry, dtype=float)

        self.model = None
        self.popt = None
        self.pcov = None

    def fit(self, model, p0=None, order=1):
        '''
            Fit a function to the data.
            mode: 'poly' if polyonmial, otherwise a function
            p0: initial parameters
            order: order of the polynomial (if mode is 'poly')
            returns: fitted parameters, pcov
        '''
        if model == 'poly':
            self._polyfit(order)
        else:
            self._curve_fit(model, p0)
        return self.popt, self.pcov


    def _curve_fit(self, func, p0=None):
        '''
            Fit a function to the data.
            func: function to fit
            p0: initial guess
            Returns: fitted parameters
        '''
        self.model = func
        self.popt, self.pcov = curve_fit(func, self.x, self.y, p0)
        return self.popt, self.pcov

    def _polyfit(self, order):
        '''
            Fit a polynomial of order 'order' to the data.
            Return the coefficients of the polynomial.
        '''
        if order == 0:
            self.model = lambda x, l: np.polyval([l, 0], x)
            self.popt, self.pcov = self.curve_fit(self.model)
        else:
            self.model = lambda x, *p: np.polyval(p, x)
            self.popt, self.pcov = np.polyfit(self.x, self.y, order, cov=True)
        return self.popt, self.pcov

    @property
    def model_param_errors(self):
        '''
            Return the errors of the fitted parameters.
        '''
        if self.pcov is None:
            raise ValueError('No fitted parameters')
        return np.sqrt(np.diag(self.pcov))

    def chi2(self, manualmethod=True, verbose=False):
        '''
            Return the chi2 and p_value of the fit.
            tutorial: https://stattrek.com/chi-square-test/goodness-of-fit.aspx
            p_value chico es mejor
        '''
        if self.model is None:
            raise ValueError('No fitted function')
        if self.popt is None:
            raise ValueError('No fitted parameters')

        if manualmethod:
            fity = self.fitfunc(self.x)
            chi_sqr = np.sum((self.y - fity)**2 / self.erry**2)
            p_chi = chi2.sf(chi_sqr, len(self.y) - 1 - len(self.popt))
        else:
            chi_sqr, p_chi = chisquare(self.y, np.array(self.fitfunc(self.x)), len(self.y) - 1 - len(self.popt))
        
        if verbose:
            print('chi2:', chi_sqr)
            print('p_chi:', p_chi)
            if p_chi < 0.05:
                print('No se puede rechazar la hipotesis de que el modelo ajusta los datos')
            else:
                print('Se rechaza la hipotesis de que el modelo ajusta los datos')

        return chi_sqr, p_chi

    def fitfunc(self, x):
        '''
            Return the fitted function.
        '''
        if self.model is None:
            raise ValueError('No fitted function')
        if self.popt is None:
            raise ValueError('No fitted parameters')
        if isinstance(x, (np.ndarray, list, tuple)):
            return np.array(self.model(x, *self.popt))
        return self.model(x, *self.popt)

    def plot(self, errorbars=False, show_fit=True, show=True):
        '''
            Plot the data and the fitted function.
        '''
        if self.model is None:
            raise ValueError('No fitted function')
        if self.popt is None:
            raise ValueError('No fitted parameters')
        if errorbars:
            if self.errx is None and self.erry is None:
                raise ValueError('No errorbars')
            elif self.errx is None:
                plt.errorbar(self.x, self.y, yerr=self.erry, fmt='o')
            elif self.erry is None:
                plt.errorbar(self.x, self.y, xerr=self.errx, fmt='o')
            else:
                plt.errorbar(self.x, self.y, xerr=self.errx, yerr=self.erry, fmt='o')
        else:
            plt.scatter(self.x, self.y, label='data')
        if show_fit:
            plt.plot(self.x, self.fitfunc(self.x), label='fit')
        plt.legend()
        if show:
            plt.show()

def help():
    print('add documentation, the class is \'Model\'')

if __name__ == '__main__':
    x = np.linspace(0, 10, 100)
    y = 2 + 5 * x + np.random.normal(0, 0.1, len(x))
    def fit(x, m, b):
        return m * x + b
    yerr = np.array([0.1 for _ in range(len(y))])

    model = Model(x, y, erry=yerr)
    model.fit('poly', order=1)
    model.plot(errorbars=True)
