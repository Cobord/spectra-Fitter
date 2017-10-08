import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sklearn.mixture
import math

# Define some test data which is close to Gaussian
data = np.random.normal(loc=-4.0,scale=6.0,size=10000)+np.random.normal(loc=1.0,scale=1.0,size=10000)
#data = numpy.random.normal(size=10000)*5.+10

hist, bin_edges = np.histogram(data, density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1., 0., 1.]

coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

# Get the fitted curve
hist_fit = gauss(bin_centres, *coeff)

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Fitted normalization = ', coeff[0])
print('Fitted mean = ', coeff[1])
print('Fitted standard deviation = ', coeff[2])

#fit with GMM
gmm = sklearn.mixture.GaussianMixture(n_components=1,covariance_type='full',tol=.00001,n_init=1,
            means_init=[[coeff[1]]],precisions_init=[[[coeff[2]]]])
r = gmm.fit(data[:, np.newaxis])
coeff = (r.means_[0, 0], np.sqrt(r.covariances_[0, 0, 0]))
normalization = 1./(coeff[1]*np.sqrt(2*math.pi))
coeff = (normalization,coeff[0],coeff[1])
hist_fit2 = gauss(bin_centres,*coeff)

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Fitted normalization = ', coeff[0])
print('Fitted mean = ', coeff[1])
print('Fitted standard deviation = ', coeff[2])

plt.plot(bin_centres, hist, 'b-',label='Test data')
plt.plot(bin_centres, hist_fit, 'r--', label='Fitted data w curve_fit')
plt.plot(bin_centres, hist_fit2, 'y--', label='Fitted data w gmm')
plt.legend(loc='best')

plt.show()
plt.close()