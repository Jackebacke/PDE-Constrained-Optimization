import poisson_discretization_DpDm as pd
import numpy as np

def synthetic_inversion(small_grid=False):
    """Setup for inversion with the heat equation"""

    config_dict = {}

    # Domain and final time
    Tend = 1
    lim_x = [-1, 1]
    lim_y = [-1, 1]
    
    # Material parameter b: Disk-shaped discontinuity, jump from 10 to 1
    b_true_fun = lambda x, y: 10-9*np.heaviside(0.5**2 - (x**2 + y**2), 0*x)
    b_initial_fun = lambda x, y: 0.0*x + 2.0

    # Boundary conditions
    bc_opts = {
        'type': 'robin',
        'a': 1 
    }

    # Discretization parameters
    mx = 41
    my = 41
    order = 5
    dt = 1e-2
    if small_grid:
        mx = 11
        my = 11

    # Create grid
    grid = pd.create_grid(mx, my, lim_x, lim_y)
    Xv = grid['Xv']
    Yv = grid['Yv']

    # Evaluate on grid
    b_true = b_true_fun(Xv, Yv)
    b_initial = b_initial_fun(Xv, Yv)

    # Zero initial data
    sol0 = 0.0*Xv

    # Point source
    sbp_discr = pd.assemble_matrices(grid, order) # Dummy
    H = sbp_discr['H']
    x_s = 0.0
    y_s = 0.0
    t0 = 0.3
    sigma = 0.02
    g = lambda t: 1e3*np.exp(-0.5*(t-t0)**2/sigma**2)
    d = discrete_delta_fun(x_s, y_s, grid, H)
    source_fun = lambda t: g(t) * d

    config_dict['grid'] = grid
    config_dict['Tend'] = Tend
    config_dict['y0'] = sol0
    config_dict['dt'] = dt
    config_dict['b_true'] = b_true
    config_dict['b_initial'] = b_initial
    config_dict['source_fun'] = source_fun
    config_dict['bc_opts'] = bc_opts
    config_dict['order'] = order

    return config_dict

def closest_index(x, y, grid):
    """Helper function for discrete point sources"""
    X = grid['Xv']
    Y = grid['Yv']
    dist = (X-x)**2 + (Y-y)**2
    index = np.argmin(dist)
    return index

def discrete_delta_fun(x, y, grid, H):
    """Return discrete delta function (point source)"""
    hx = grid['hx']
    hy = grid['hy']

    discr_delta = 0.0*grid['Xv']
    index = closest_index(x, y, grid)

    # Note scaling with 1/(hx*hy) to approximate delta function
    discr_delta[index] = 1/H[index, index]

    return discr_delta