import numpy as np
import poisson_discretization_DpDm as pd
import scipy.sparse.linalg as spsplg
from scipy.sparse import diags_array


def setup():
    """Setup for the Schrodinger problem"""

    # Intitial guess for b: constant = 1
    b_initial_fun = lambda x, y: 0.0 * x + 1.0

    # Boundary conditions
    bc_opts = {"type": "robin", "a": 1}

    # Domain and discretization parameters
    lim_x = [-1, 1]
    lim_y = [-1, 1]
    mx = 41
    my = 41
    order = 5

    # Create grid
    grid = pd.create_grid(mx, my, lim_x, lim_y)
    Xv = grid["Xv"]
    Yv = grid["Yv"]

    # Background potential q0: Constant = 1
    q0 = 1.0 + 0 * Xv

    # True q: three Gaussian patches
    A = 0.5
    q = (
        1
        + A * pd.shifted_gaussian(Xv, Yv, 0.5, 0.4, 0.1)
        + A * pd.shifted_gaussian(Xv, Yv, 0.5, -0.4, 0.1)
        + A * pd.shifted_gaussian(Xv, Yv, -0.5, 0.0, 0.1)
    )

    # Known source u
    x0 = 0.0
    y0 = 0.0
    width = 0.2
    u = -1e3 * pd.shifted_gaussian(Xv, Yv, x0, y0, width)

    # Solve with true q to get synthetic data
    sbp_ops = pd.assemble_matrices(grid, order, bc_opts=bc_opts)
    D_true = sbp_ops["Laplace_with_BC"] + diags_array(q)
    y_data = spsplg.spsolve(D_true, u)

    # Dict with everything
    config_dict = {}
    config_dict["y_data"] = y_data
    config_dict["u"] = u
    config_dict["a"] = bc_opts["a"]
    config_dict["q0"] = q0
    config_dict["q"] = q

    config_dict["sbp_ops"] = sbp_ops
    config_dict["grid"] = grid

    return config_dict
