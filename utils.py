import numpy as np
from scipy.interpolate import CubicHermiteSpline, interp1d
from kernels import Wendland, Matern


def points(step1, step2=None):
    if not step2:
        step2 = step1
        
    x1 = np.arange(0, 1 + step1, step1)
    x1 = x1[x1 <= 1]
    if np.max(x1) < 1:
        x1 = np.r_[x1, 1]

    x2 = np.arange(0, 1 + step2, step2)
    x2 = x2[x2 <= 1]
    if np.max(x2) < 1:
        x2 = np.r_[x2, 1]

    xx, yy = np.meshgrid(x1, x2)

    X = np.c_[xx.flatten(), yy.flatten()]
    return X, x1, x2


def sigma(f, g):
    return np.sum((f - f.mean()) * (g - g.mean())) / len(f)


def S(f, g, c=1):
   return (2 * sigma(f, g) + c) / (sigma(f, f) + sigma(g, g) + c)


def M(f, g, c=1):
   fm = f.mean()
   gm = g.mean()
   return (2 * fm * gm + c) / (fm ** 2 + gm ** 2 + c)


def DSSIM(f, g, c=1):
    return 1 - M(f, g, c) * S(f, g, c)


def cf(f,c=1e-5):
    return 4 / (sigma(f,f) + c) + 1 / (f.mean() ** 2 + c)

    
def cf_g(f, g, c=1e-5):
    return 4 / (sigma(f, f) + sigma(g, g) + c) + 1 / (f.mean() ** 2 + g.mean() ** 2 + c)
   

def divide_(a, b):
    if b.all() == 0:
        return np.nan
    else: 
        return a/b


def divide(a, b):
    c = np.zeros(b.shape)
    for i in range(0, b.shape[0]):
        if b[i] == 0:
            c[i] = np.nan
        else: 
            c[i] = a[i]/b[i]
    return c


def get_test_function(test_case=0):
    
    if test_case == 0:
        f = lambda x: 2 * (x[:, 0] * x[:, 1]) ** 2 - np.sinc(x[:, 0]) * np.sinc(x[:, 1])
        fx = lambda x: 4 * x[:, 0] * x[:, 1] ** 2 - np.sinc(x[:, 1]) * np.nan_to_num(divide(np.pi * x[:, 0] * np.cos(np.pi * x[:, 0]) - np.sin(np.pi*x[:, 0]),np.pi * x[:, 0]**2))
        fy = lambda x: 4 * x[:, 1] * x[:, 0] ** 2 - np.sinc(x[:, 0]) * np.nan_to_num(divide(np.pi * x[:, 1] * np.cos(np.pi * x[:, 1]) - np.sin(np.pi*x[:, 1]),np.pi * x[:, 1]**2))
    
    elif test_case == 1:
        f = lambda x: np.exp(-(x[:, 0] + x[:, 1])) - 3 * x[:, 0] + x[:, 1] + 5
        fx = lambda x: -np.exp(-(x[:, 0] + x[:, 1])) - 3 
        fy = lambda x: -np.exp(-(x[:, 0] + x[:, 1])) + 1
  
    return f, fx, fy


def bilinear_interp(X, x_nodes, x_test, f):
    
    N_1d = len(x_nodes)
    N_evx = len(x_test)
    
    x_slices = np.zeros((N_1d, N_evx))
    
    y = f(X)
    y = np.reshape(y, (N_1d, N_1d))
    
    for i in range(N_1d):
        int_1d = interp1d(x_nodes, y[i, :])
        x_slices[i,:] = int_1d(x_test)
    
    interp_values = np.zeros((N_evx, N_evx))
    for i in range(N_evx):
        tmp = interp1d(x_nodes, x_slices[:,i])
        interp_values[:, i] = tmp(x_test)
    val = interp_values.ravel()
    
    return val
    

def bicubic_interp(x_nodes, f, fy, x_test, fx):
        
    N_1d = len(x_nodes)
    N_evx = len(x_test)
    
    # Fit a cubic in the y direction
    val_x = np.zeros((N_1d, N_evx))    
    for i in range(N_1d):
        # 1d grid in the y direction, with a fixed x (an interpolation point)
        X_local = np.c_[x_nodes[i] * np.ones((N_1d, 1)), x_nodes]
        # Evaluation of the function on the int. points
        y_local = f(X_local)
        # Evaluation of the derivative on the int. points
        dy_local = fy(X_local)
        # Cubic intepolant
        tmp = CubicHermiteSpline(x_nodes, y_local, dy_local)
        # Evaluation on the test set
        val_x[i, :] = tmp(x_test)
    
    
    # Fit a second cubic in the x direction
    interp_values = np.zeros((N_evx, N_evx))
    for j in range(N_evx):
        # 1d grid in the x direction, with a fixed x (a test point)
        X_local = np.c_[x_nodes, x_test[j] * np.ones((N_1d, 1))]
        # Evaluation of the function on the int. points
        y_local = val_x[:, j]
        # Evaluation of the derivative on the int. points
        dx_local = fx(X_local)
        # dx_local = (y_local[1:] - y_local[:-1]) / (y_local[1] - y_local[0])
        # dx_local = np.r_[dx_local, dx_local[-1]]
        # Cubic intepolant
        tmp = CubicHermiteSpline(x_nodes, y_local, dx_local)
        # Evaluation on the test set
        interp_values[j, :] = tmp(x_test)
    
    # Move the values to a unique vector
    val = interp_values.ravel()
    
    return val


def kernel_interp(X, y, X_test, kernel_type):
    if kernel_type == 'wendland':
        kernel = Wendland(ep=1, k=1, d=2)
    elif kernel_type == 'matern':
        kernel = Matern(ep=1, k=3)
 
    a = np.linalg.solve(kernel.eval(X, X), y)
    val = kernel.eval(X_test, X) @ a
    return val
    