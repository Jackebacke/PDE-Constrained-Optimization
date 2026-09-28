import poisson_discretization_DpDm as pd
import setup_heat as setup
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplg
import numpy as np
import matplotlib.pyplot as plt

def plot_setup(setup_fun=setup.synthetic_inversion):

    # Load problem setup
    config_dict = setup_fun()
    grid = config_dict['grid']
    y0 = config_dict['y0']
    b_true = config_dict['b_true']
    b_intial = config_dict['b_initial']

    # Plot initial data y0 and material parameter b
    fig, axes = subplots_2d(grid, [y0, b_true, b_intial])
    axes[0].set_title('Initial data y0')
    axes[1].set_title('True material parameter b')
    axes[2].set_title('Initial guess for material parameter b')
    plt.tight_layout()

def subplots_2d(grid, fields, sz=None):
    """Helper function that plots several 2d plots in one figure.
    X, Y: Coordinate arrays (from np.meshgrid)
    fields: list of fields to plot. One subplot per field.
    sz: subplot arrangement, e.g. (2, 3) for 2 rows, three columns.
    """
    X = grid['X']
    Y = grid['Y']
    fields = [np.reshape(f, X.shape) for f in fields]
    n_fields = len(fields)
    if sz is None:
        fig, axes = plt.subplots(n_fields, 1)
    else:
        fig, axes = plt.subplots(*sz)
        axes = axes.flatten()
    im = []
    for i in range(n_fields):
        im.append(axes[i].pcolormesh(X, Y, fields[i], shading='auto'))
        fig.colorbar(im[i])
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
    return fig, axes

def main():

    # --- Plot config --- 
    plot_setup()    

    plt.show()

if __name__ == '__main__':
    main()