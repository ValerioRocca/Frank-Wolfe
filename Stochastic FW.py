
# ------ SUMMARY ------
# In constraint optimization, the complexity lies in both 1) handling the
# objective function and 2) handling the constraint set.
# Stochastic FW alleviate both computational burdens:
# - Objective function's burden by using approximate first-order information;
# - Constraint set's budern by maintaining feasibility without using projections.
# In this paper (Combettes et al.) we improve the quality of SFW methods by
# improving the quality of their first-order information, by blending in them
# adaptive gradients.
# FUNCTIONS:
# - FW: the known one
# - SFW (Stochastic FW): replaces the gradient with a stochastic estimator
# - SVRF (Stochastic Variance-Reduced FW): integrates variance reduction in the
#   estimation of the stochastic gradients to improve batch-size rate to
#   bt = O(t)

import numpy as np
import time
import sklearn

from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------
#      GENERAL FUNCTIONS
# ---------------------------

# FW solves linear programs, which are specific optimization problems in which both
# the objective function and the constraints are linear fnctions of the decision
# variables.
# To use FW for a regression task we consider Lasso regression, which can be
# formulated as a linear program:
# - Loss function: |Ax - y|^2
#    - Gradient: 2* A(Ax - y)
# - Constraint: { x : |x|_1 <= t }, where the value of t is determined by the 
#   hyperparameter lambda
# CHECK SE IL GRADIENTE DELLA LASSO E' CALCOLATO CORRETTAMENTE

def lasso_gradient(xk, sample, y_batch):
    # grad = (1/m) * X^T * (X * xk - y_true) + lambda * sign(xk)
    grad = (1/sample.shape[0]) * np.matmul(sample.T, (np.matmul(sample,xk) - y_batch)) \
        + lamb*np.sign(xk)
    return grad

def lasso_argmin(grad):
    # Initialize the zero vector
    xk_hat = np.zeros(len(grad))

    # Check if all coordinates of x are 0
    # If they are, then the Oracle contains zero vector
    if (grad != 0).sum() == 0:
        return xk_hat

    # Otherwise, follow the following steps
    else:
        # Compute the (element-wise) absolute value of x
        a = abs(grad)
        # Find the first coordinate that has the maximum absolute value
        i = np.nonzero(a == max(a))[0][0]
        # Compute s
        xk_hat[i] = - np.sign(grad[i]) * lamb
        return xk_hat
    


# --------------------------
#      SFW ALGORITHM
# --------------------------

# ISPIRATA DA:
# https://github.com/paulmelki/Frank-Wolfe-Algorithm-Python/blob/master/frank_wolfe.py

# IMPORTANTE: Xk SONO I PESI. LE OSSERVAZIONI SONO "SAMPLES"

def SFW(x1, sample, y_true, batch_dim, eps=1e-6, epochs=100):

    # 1. Choose the starting point
    xk = x1

    # NA. Add a column of 1 in front of the sample (to compute the gradient of beta0)
    sample_1 = np.concatenate((np.ones((sample.shape[0], 1)), sample), axis=1)

    for k in range(epochs):

        print(f"Iteration {k} of SFW")

        # NA. Extract a batch of dimension "batch_dim" from the sample
        indices = np.random.choice(sample_1.shape[0], batch_dim, replace=False)
        y_batch = y_true[indices]
        sample_batch = sample_1[indices]

        # 3. Compute the search direction
        # Compute the gradient
        grad = lasso_gradient(xk, sample_batch, y_batch)
        xk_hat = lasso_argmin(grad)
        
        # 4. Check the stopping criterion on xk_hat
        if k > 1:
            if np.matmul(grad.T,(xk_hat - xk)) >= -eps: break

        # NA. Compute the step size
        alpha_k = 2.0/(k+1)      

        # 5. Update the iterate
        xk =  xk + alpha_k*(xk_hat - xk)

    return xk

# --------------------------
#      MAIN
# --------------------------

def SFW_predict(xk, X_test):
    test_1 = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
    return np.matmul(test_1,xk)

# Initialisation
n_obs = 10000
n_var = 10
noise_factor = 0.1

x1 = np.zeros(n_var+1)
batch_dim = 30
lamb = 0.1

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

# Compute Lasso regression both with SFW and scikit
SFW_weights = SFW(x1, X_train, y_train, batch_dim)
SFW_pred = SFW_predict(SFW_weights, X_test)
SFW_mse = mean_squared_error(y_test, SFW_pred)

sklearn_pred = Lasso(alpha = lamb).fit(X_train,y_train).predict(X_test)
sklearn_mse = mean_squared_error(y_test, sklearn_pred)

print(f"The MSE predicted by Sklearn is: {sklearn_mse}\nThe MSE predicted by SFW is: {SFW_mse}")






