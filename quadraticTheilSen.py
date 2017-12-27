# Edited from Florian Wilhelm -- <florian.wilhelm@gmail.com>
# in the docs for scikit-learn

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline,Pipeline

#quadratic_model = make_pipeline(PolynomialFeatures(2),Ridge())
quadratic_model = Pipeline([('poly',PolynomialFeatures(2)),('linear',Ridge())])
#quadratic_RANSAC = make_pipeline(PolynomialFeatures(2),RANSACRegressor(random_state=42))
quadratic_RANSAC = Pipeline([('poly',PolynomialFeatures(2)),('linear',RANSACRegressor(random_state=42))])
quadratic_TheilSen = Pipeline([('poly',PolynomialFeatures(2)),('linear',TheilSenRegressor(random_state=42))])

estimators = [('QuadraticRANSAC', quadratic_RANSAC), 
              ('QuadraticRidge',quadratic_model),
              ('QuadraticTheilSen',quadratic_TheilSen)]
colors = {'OLS': 'turquoise','QuadraticRidge': 'turquoise', 'TheilSen': 'gold','QuadraticTheilSen': 'gold',
            'RANSAC': 'green','QuadraticRANSAC': 'green'}
lw = 2

# #############################################################################
# Outliers 5% each direction

np.random.seed(0)
n_samples = 200
# Quadratic model y = 4*x^2 + 3*x + N(2, 0.1**2)
x = np.random.randn(n_samples)
w = 3.
c = 2.
w2 = 4
noise = 0.1 * np.random.randn(n_samples)
y = w2* x **2 + w * x + c + noise
# 5% outliers
y[-20:-10] += -20 * x[-20:-10]
# 5% outliers
x[-10:] = 4
y[-10:] += 22
X = x[:, np.newaxis]

plt.scatter(x, y, color='indigo', marker='x', s=60)
line_x = np.linspace(-3,3,10)
for name, estimator in estimators:
    t0 = time.time()
    estimator.fit(X, y)
    elapsed_time = time.time() - t0
    y_pred = estimator.predict(line_x.reshape(10, 1))
    # print the coefficients
    try:
        if(name=='QuadraticRidge'):
            print('Quadratic Ridge: ')
            print(estimator.named_steps['linear'].coef_)
        elif(name=='QuadraticRANSAC'):
            print('QuadraticRANSAC')
            print(estimator.named_steps['linear'].estimator_.coef_)
        elif(name=='QuadraticTheilSen'):
            print('QuadraticTheilSen')
            print(estimator.named_steps['linear'].coef_)
        else:
            print('NoneOfTheAbove')
    except:
        print('Error')
    plt.plot(line_x, y_pred, color=colors[name], linewidth=lw,
             label='%s (fit time: %.2fs)' % (name, elapsed_time))

plt.axis('tight')
plt.legend(loc='best')
plt.title("Corrupt Quadratic")