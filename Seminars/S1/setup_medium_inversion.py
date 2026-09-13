import numpy as np
import poisson_discretization_DpDm as pd
import scipy.sparse.linalg as spsplg

def problem_setup(mx=21, my=21):
    """Setup for the Poisson medium inversion problem"""

    # True b: Disk-shaped discontinuity, jump from 1 to 10
    b_true_fun = lambda x, y: 9*np.heaviside(0.5**2 - (x**2 + y**2), 0*x) + 1.0

    # Intitial guess for b: constant = 1
    b_initial_fun = lambda x, y: 0.0*x + 1.0

    # Boundary conditions
    bc_opts = {
        'type': 'robin',
        'a': 1 
    }

    # Domain and discretization parameters
    lim_x = [-1, 1]
    lim_y = [-1, 1]
    order = 5

    # Create grid
    grid = pd.create_grid(mx, my, lim_x, lim_y)
    Xv = grid['Xv']
    Yv = grid['Yv']

    # Evaluate variable coefficient on grid
    b_true = b_true_fun(Xv, Yv)
    b_initial = b_initial_fun(Xv, Yv)

    # Known source u
    x0 = 0.5
    y0 = 0
    width = 0.2
    u = 1e3*pd.shifted_gaussian(Xv, Yv, x0, y0, width)

    # Solve with true b to get synthetic data
    matrices_true = pd.assemble_matrices(grid, order, b_true, bc_opts)
    D_true = matrices_true['Laplace_with_BC']
    y_data = spsplg.spsolve(D_true, u)

    # Dict of initial guesses for parameter(s) that we are inverting for
    initial_parameters = {}
    initial_parameters['b'] = b_initial

    # Dict of known parameters, including data
    true_parameters = {}
    true_parameters['y_data'] = y_data
    true_parameters['u'] = u
    true_parameters['b'] = b_true
    true_parameters['a'] = bc_opts['a']
    
    return grid, bc_opts, initial_parameters, true_parameters, order