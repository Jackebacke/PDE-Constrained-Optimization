import poisson_discretization_DpDm as pd

def problem_setup():
    """Defines parameters for the Poisson source inversion problem"""

    # Domain and discretization parameters
    lim_x = [-1, 1]
    lim_y = [-1, 1]
    mx = 41
    my = 41
    order = 5

    # Create grid
    grid = pd.create_grid(mx, my, lim_x, lim_y)
    Xv = grid['Xv']
    Yv = grid['Yv']

    # Build SBP-SAT matrices
    sbp_discr = pd.assemble_matrices(grid, order)

    # Setup target state z
    x0 = 0
    y0 = 0
    width = 0.2
    z = pd.shifted_gaussian(Xv, Yv, x0, y0, width)

    # Initial guess for u
    u0 = 10*pd.shifted_gaussian(Xv, Yv, x0-0.5, y0-0.5, width)

    # Regularization parameter
    alpha = 1e-4

    return sbp_discr, grid, z, u0, alpha